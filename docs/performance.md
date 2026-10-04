# Single-GPU serving performance

## October 4, 2026 adaptive-MTP/compression qualification

The PR candidate is based on `master` at `87f06898`. Qualification uses
`models/qwen3_8_27b_nvfp4.v3.ninfer` on an RTX 5090 Laptop GPU (24 GiB), CUDA 13.1.115,
GCC 15.3, Release/`sm_120a`. This artifact combines NVFP4 FFN weights with FP8 GDN output
weights; its name does not imply that every projection is NVFP4.

A short C=2/K15 greedy check initially diverged from MTP-off at token 26, identically with
compression on/off and with Graphs/eager execution. Fixed K3 and K7 passed. Intermediate
inspection localized the first difference to FP8 A8 LinearAdd with identical inputs: full
versus partial token tiles permitted different MUL+ADD contraction. Explicit FP32 activation
scaling followed by the residual FMA fixes that boundary without changing schedules or workspace.
An A16 reduction-schedule candidate did not address this A8 route and was removed.

The regression checks both real residual shapes at T=32/128 against ordinary columns and an
independent FP64 oracle. It fails against master and passes the candidate. Unit FP8 codes isolate
the cancellation case's epilogue rounding; an earlier full-range cancellation fixture exceeded
the extra A16-tight represented-A8 criterion because of FP32 dot accumulation error, while passing
the public A8 criterion. The final cancellation fixture keeps the dot exact rather than relaxing
its criterion. Normal and wide-exponent inputs retain the full-range packed-weight oracle.

Nineteen focused tests passed: codec exact decoding/guards/Graph replay, loading and binding,
NVFP4 Linear/LinearAdd/SwiGLU, FP8 A8 Linear/LinearAdd/SwiGLU and input projections, replay state,
MTP control, CLI/server options, OpenAI schema and request logging. Test review also corrected
a stale artifact-object handle, restored compact A16 cancellation coverage, fixed its zero-size
workspace fixture, and aligned the real test's logical context ceiling with its rounded KV
reservation. No tolerances were relaxed for the restored NVFP4 coverage.

Real Engine qualification passed two fresh repeats against MTP-off for:

- C=2 adaptive K3/K7/K15, compression enabled, prefix reuse, 128/127 output tokens, eager
  execution on all six KV formats (BF16, INT8, FP8, NVFP4, K8V4, KVarN).
- C=2 adaptive K15 maximum, compressed KVarN, CUDA Graphs and prefix reuse: short prompts with
  256/255 output tokens and ragged 8,190/8,183-token prompts with 256/255 output tokens.
- C=8 fixed K15, compressed KVarN, CUDA Graphs, short prompts and ragged 256..249 output budgets,
  with context retention disabled.

Capacity failures remain distinct from arithmetic qualification. On this 24 GiB device, the
C8/K15 16K-context corpus fixture could not reserve runtime memory, even without retention
(4,861,090,560 bytes required versus 3,937,861,632 available). Short-context C8/K15 with eight
retained checkpoint slots also failed reservation, including prefill-chunk 128
(3,967,889,152 versus 3,937,861,632 bytes). Cache-disabled short-context execution passed;
this does not qualify those larger configurations. DFlash/DFlash2, Vision generation, and other
real artifacts were not requalified in this follow-up.

### Matched public Engine measurements

A public-Engine-only batch driver prepares two corpus slices (offsets 0/128), submits both before
waiting, and requests 512 greedy tokens each. Sampling penalties, prefix reuse, retention and
model stop tokens are disabled; raw output, KVarN, optimized proposal head, CUDA Graphs and
2,048-token prefill chunks are enabled. Context capacity is prompt length + 1,536 per request,
with shared capacity twice that. Each configuration uses one discarded warmup and three measured
batches, run sequentially without other GPU workloads. Startup and token-file output are excluded
from batch timing. Decode rate divides committed tokens by the difference of Engine decode
Host-active plus device-wait counters, not summed overlapping request durations.

The baseline is current master `87f06898`, fixed K3, compression disabled. Candidate fixed K3
isolates integration overhead; adaptive K15 is a new capability, not an optimized-master K15
comparison. Compression compares against the same candidate adaptive configuration. Results are
arithmetic means of three batches; brackets give the full observed decode-rate range. Every
measured output token sequence matched the corresponding master sequence exactly.

| Prompt/request | Configuration | Decode tok/s [min,max] | Batch seconds | Mean physical batch |
|---:|---|---:|---:|---:|
| 2,048 | Master K3 | 237.43 [237.33,237.49] | 5.1803 | 1.985 |
| 2,048 | Candidate K3 | 237.55 [237.52,237.59] | 5.1810 | 1.985 |
| 2,048 | Candidate adaptive K15 | 350.69 [350.18,351.11] | 3.7941 | 1.794 |
| 2,048 | Adaptive + compression | 346.17 [345.95,346.54] | 3.8369 | 1.794 |
| 8,192 | Master K3 | 237.58 [237.55,237.62] | 8.0773 | 1.940 |
| 8,192 | Candidate K3 | 237.39 [237.25,237.50] | 8.0864 | 1.940 |
| 8,192 | Candidate adaptive K15 | 474.10 [473.77,474.28] | 5.9444 | 1.864 |
| 8,192 | Adaptive + compression | 468.47 [468.06,468.72] | 5.9844 | 1.864 |

Fixed-K3 decode changes are +0.05%/-0.08%, while batch latency changes are +0.01%/+0.11% at
2K/8K; no material fixed-K3 improvement is claimed. Adaptive decode improves 47.7%/99.6% versus
master K3, with batch latency down 26.8%/26.4%. Compression costs 1.29%/1.19% decode throughput
and adds 1.13%/0.67% batch latency versus uncompressed adaptive execution. These claims cover
these two corpus workloads and C=2 only.

Compression reduces weight capacity from 21,183,929,344 to 20,781,619,200 bytes (383.67 MiB).
Adaptive K15 adds 52.76 MiB of runtime reservation versus candidate K3; compression adds no
further reservation in these profiles because existing peak workspace already accommodates
expansion. Candidate K3 uses 1,536 more runtime bytes than master. Individual observed startup
times were 5.15–5.17 seconds uncompressed and 7.61–7.62 seconds compressed (about +2.45 seconds);
these are single startup observations, not a startup-latency distribution. Runtime reservations
were 806,183,168/861,503,744 bytes for K3/K15 at 2K and 1,153,623,296/1,208,943,872 at 8K.

### Public Op measurements and exceptions

Cold CUDA Graph benchmarks use 10 warmups, 100 samples and a 256 MiB L2 flush per sample.
For scale expansion, the baseline is the previous compressed decoder at `99d1be8f`, built with
the same benchmark driver; it is not master, which has no compression Op. All nine cases are
shown below. High-entropy inputs are decoder stress cases: the loader normally leaves such
weights uncompressed. Alignment increases compressed residency by 0.18–1.35% in this matrix.

| N/K | Symbols | Old median µs | New median µs | Time change |
|---|---:|---:|---:|---:|
| 5120/6144 | 15 | 8.224 | 6.144 | −25.3% |
| 5120/6144 | 17 | 10.240 | 8.192 | −20.0% |
| 5120/6144 | 256 | 12.288 | 10.240 | −16.7% |
| 5120/17408 | 15 | 14.368 | 10.240 | −28.7% |
| 5120/17408 | 17 | 16.448 | 16.288 | −1.0% |
| 5120/17408 | 256 | 22.560 | 20.480 | −9.2% |
| 34816/5120 | 15 | 24.576 | 20.416 | −16.9% |
| 34816/5120 | 17 | 26.688 | 26.656 | −0.1% |
| 34816/5120 | 256 | 36.928 | 34.816 | −5.7% |

FP8 A8 LinearAdd compares against master on N=5120, K=6144/17408 and
T=1,7,16,31,32,33,64,65,127,128. An initial pass showed K=17408 slowdowns of 11.3–30.9%; a
reversed-order spot check and two full reversed-order confirmation sweeps did not reproduce
them. Their cause remains unresolved; they are not dismissed as proven timing noise. The two
confirmation sweeps give the following per-case ranges of candidate median time change:

| T | K=6144 | K=17408 |
|---:|---:|---:|
| 1 | −0.30..−0.06% | −0.03..0.00% |
| 7 | −0.07..−0.06% | −1.79..0.00% |
| 16 | 0.00% | 0.00% |
| 31 | −0.07..0.00% | 0.00..+0.03% |
| 32 | −0.14..−0.06% | −0.06..+0.08% |
| 33 | 0.00% | −0.02..+0.06% |
| 64 | −0.07..−0.06% | −0.03..0.00% |
| 65 | +0.22..+2.63% | 0.00..+0.97% |
| 127 | +2.46..+3.33% | −1.19..+0.96% |
| 128 | −0.05..0.00% | −1.15..0.00% |

The first K=6144 pass ranged from −1.32% to +3.63% (worst at T=65). T=127's +3.33% initial
cost reproduced in confirmation; the correctness fix retains this small Op-level regression.
Neither Op timings nor the historical decoder results below establish a general serving gain.

## NVFP4 compressed-scale expansion, concurrency two

The September 26, 2026 comparison uses Qwen3.8-27B QUASAR NVFP4 v3 on the RTX 5090 Laptop
GPU (24 GiB), driver 595.91.07, CUDA 13.1.115, Release/`sm_120a`. Compressed records now follow
native swizzled byte order and are 16-byte aligned. The decoder expands two tiles concurrently
per block, uses vector memory accesses and four-byte palette lookups, and scans only present
literal escapes. Expansion remains lossless and uses the existing caller-owned workspace.

Two public Engine requests are prepared before timing and submitted together. Each generates
512 greedy tokens from `bench/fixtures/bench_corpus.ids`, with prompt offsets 0 and 128,
prefix reuse and model stop tokens disabled, 2,048-token prefill chunks, KVarN K4V2-G128 KV,
optimized proposal head and CUDA Graphs. Per-request context capacity is prompt length + 1,536;
shared KV capacity is twice that. Each run discards one warmup and measures three batches.
The adaptive comparison repeats both the old and new decoder to check timing drift; its entries
average six measured batches per decoder. K3 uses three measured batches per decoder.

| Prompt tokens/request | MTP policy | Old compressed decode (tok/s) | New compressed decode (tok/s) | Gain | Old batch (s) | New batch (s) |
|---:|---|---:|---:|---:|---:|---:|
| 8,192 | Adaptive K3/K7/K15 | 407.6 | 412.4 | 1.2% | 5.432 | 5.384 |
| 2,048 | Fixed K3 | 238.1 | 240.1 | 0.8% | 4.954 | 4.920 |

Decode throughput uses committed tokens divided by Engine decode Host-active plus device-wait
interval counters, without summing overlapping request times. Mean physical batch sizes are
1.84 and 1.99. These are small workload-specific improvements. The uncompressed 8K adaptive
control measured 415.6 tok/s and 5.374 s/batch: compression still costs about 0.8% of decode
throughput on this workload. All generated token IDs matched between old/new compression and
the uncompressed control; fixed-K3 old/new outputs also matched exactly.

The [scale-expansion Op benchmark](../bench/README.md#nvfp4-scale-expansion) measures cold
CUDA Graph replay separately. At N=5120/K=17408, median expansion time changed from 16.38 to
10.30 microseconds for 15-symbol inputs and from 18.46 to 16.42 microseconds for 17-symbol
inputs. At N=34816/K=5120 it changed from 26.66 to 22.46 microseconds for 15-symbol inputs;
the 17-symbol case was effectively unchanged at about 28.8 microseconds. These synthetic
distributions establish decoder behavior, not an equivalent inference speedup.

Qualification passed the independent byte-wise codec oracle and exact native-plane comparisons
at real FFN shapes, including escapes, partial launch tails, output guards and repeated Graph
replay. The loading/binding test and independent-oracle NVFP4 Linear A16/A4, LinearAdd and
LinearSwiGLU suites also passed. Record alignment adds a small amount of resident padding;
the `.ninfer` artifact and mathematical scale bytes are unchanged.
The full serving configuration also reached ready with 401,408 KV tokens, concurrency two,
Vision, adaptive MTP and 8 GiB Host KV: runtime reservation was 7.12 GiB and free VRAM was
220.1 MiB. After shutdown, idle GPU usage returned to 31 MiB.

## RTX 5090 Laptop adaptive-MTP concurrency-two tuning

The September 21, 2026 comparison uses Qwen3.8-27B QUASAR NVFP4 v3 on the RTX 5090 Laptop
GPU (24 GiB), CUDA 13.1.115, GCC 15.3, Release/`sm_120a`. Both builds use the same measured
K3/K7/K15 adaptive controller, optimized proposal head, KVarN K4V2-G128 KV and CUDA Graphs.
Only the 32-column, N=5120/K=6144 A16 residual-projection schedule changes: four output rows
per warp and eight-column tiles replace two rows and a full 32-column tile. Each column keeps
the same FMA and reduction order.

Two public Engine requests are prepared before timing and submitted together. Each generates
512 greedy tokens from `bench/fixtures/bench_corpus.ids`, with prompt offsets 0 and 128,
prefix reuse disabled, model stop tokens disabled and 2,048-token prefill chunks. Per-request
context capacity is prompt length + 1,536; shared KV capacity is twice that. Each run discards
one warmup and measures three batches. The 8K baseline was repeated after the candidate to
check the small improvement against timing drift; its table entry averages both baseline runs.

Aggregate decode throughput is the sum of committed decode tokens divided by the Engine's
decode Host-active plus device-wait interval counters. Concurrent requests' exposed decode
times are not summed. Batch wall time includes both prefills and completion of both requests.

| Prompt tokens/request | Before aggregate decode (tok/s) | After (tok/s) | Decode gain | Before batch (s) | After batch (s) |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 386.2 | 396.8 | 2.7% | 5.586 | 5.510 |
| 32,768 | 279.5 | 281.9 | 0.9% | 20.403 | 20.455 |

These include one-row decode while the second request prefills and after the first finishes;
mean physical batch sizes were 1.84 and 1.42, respectively. The 32K batch wall-time difference
is dominated by prefill variation; this measurement establishes no total-request speedup there.
The gain depends on residency at K15 with two active rows and is not a general serving-rate claim.

Nsight Systems attributed about 17% of the 8K decode interval to this projection. Targeted
Nsight Compute profiling reduced kernel time from 246 to 197 microseconds, registers/thread
from 168 to 128, and increased achieved occupancy from 22.3% to 30.9%, without spilling.
The Op profiling command is:

```bash
ncu --profile-from-start off --set basic --section Occupancy --section MemoryWorkloadAnalysis \
  build/bench/ninfer_nvfp4_linear_add_bench --n 5120 --k 6144 --policy a16 \
  --t-sweep 32 --profile
```

Qualification covers the independent FP64 Op oracle, exact batched-versus-single-column
residual updates, and C=2 KVarN greedy parity against MTP-off with ragged 8K prompts,
prefix reuse, CUDA Graph replay and all three adaptive tiers.

### Follow-up: Q4 shared-memory layout and 32-column SwiGLU

A second comparison uses the residual-tiled implementation above as its baseline, on the same
hardware/toolchain and artifact. The selected changes are:

- Pad each Q4 K-split shared code row by 16 bytes. This separates the eight MMA row groups'
  shared-memory banks while preserving aligned asynchronous transfers and the arithmetic order.
  Nsight Compute measured the N=131072/K=5120/T=2 projection at 821 versus 552 microseconds;
  DRAM utilization rose from 60.2% to 90.5%, with unchanged register count and no spills.
- Use a 32-column/128-row tile for NVFP4 fused SwiGLU at T=17..32. Previously T=32 used a
  48-column/64-row tile. Cold-cache public-Op timing at T=32 improved from 143 to 135 microseconds.
  The existing BF16 gate/up boundary and SiLU calculation are preserved.

The public Engine measurements again use one discarded warmup and three measured batches,
greedy sampling, no prefix reuse and disabled model stop tokens. The corpus pair uses the same
8K/512-output workload above. Chat pairs have a 2,048-token context capacity and 1,024 output
tokens per request; aggregate decode counters include the one-row tail when one request finishes.

| Two-request workload | Before aggregate decode (tok/s) | After (tok/s) | Gain | Before batch (s) | After batch (s) |
|---|---:|---:|---:|---:|---:|
| 8K corpus pair | 393.8 | 408.9 | 3.8% | 5.559 | 5.461 |
| Snake code pair | 250.1 | 255.0 | 2.0% | 8.367 | 8.211 |
| Snake code + storage-design reasoning | 151.7 | 152.1 | 0.3% | 13.698 | 13.662 |

The mixed-pair difference is too small to claim a material improvement. Mean physical batch sizes
were approximately 1.84, 2.00 and 1.60. The chat prompts are:

- Code, thinking disabled: “Implement a complete Snake game in Python using pygame. Include
  movement, food, scoring, collision detection, a start screen, pause, restart and game over.
  Output only the complete Python code, without explanations.”
- Reasoning, thinking enabled: “Design a crash-safe concurrent key-value store in C++. Reason
  carefully about atomicity, write-ahead logging, snapshots, compaction, and recovery after power
  loss. Compare at least three designs and analyze their correctness and failure cases before
  choosing one. Do not write code yet.”

A controller candidate allowing any confident continuing row to trigger a wider probe was
rejected: mixed-pair throughput fell from 151.7 to 143.8 tok/s, about 5.2%, with substantially
more K7 work. Wider-probe eligibility alone did not improve the aggregate objective.

Independent mathematical-oracle tests passed for Q4 Linear and its affected fused consumers,
and for NVFP4 SwiGLU including the new route boundaries and graph replay. C=2 KVarN generation
matched MTP-off token-for-token at ragged 8K and 32K contexts with prefix reuse. The parity
fixture reserves independently rounded per-request KV capacity so the long-context case truly
executes compact batches rather than serializing for lack of a physical page.

Reproduce the focused public-Op measurements with:

```bash
build/bench/ninfer_linear_bench --qtype q4 --n 131072 --k 5120 --t 2 --repeat 100
build/bench/ninfer_nvfp4_linear_swiglu_bench --policy a4 --t-sweep 16,32,48 --repeat 100
```

## RTX 5090 Laptop upstream integration through 9e163eee

The September 19, 2026 integration compares the fork's `master` at `ac508b13` with the merge
through upstream `9e163eee`. Hardware: RTX 5090 Laptop GPU, 24 GiB, driver 595.84; Release,
`sm_120a`, CUDA 13.1.115 and GCC 15.3. The artifact is Qwen3.8-27B QUASAR NVFP4. The candidate
uses an offline v3 upgrade preserving stored weight values; the baseline uses the original v2
artifact. Measurements use identical prepared token IDs, so the upgraded chat template does not
change the measured prompts.

The selected NVFP4 implementation retains upstream's shape-owned tuning, token-fast TMA scheduling,
tile-wise activation-scale loads and partial-tile support, together with the fork's qualified
low-width fused projections. Two alternatives were rejected during qualification:

- Approximate SiLU in the fused TMA SwiGLU epilogue passed the Op oracle but lowered MTP acceptance
  on the 512-token, two-request workload. Batch latency rose from about 1.80 s to 2.02 s and
  speculative row-rounds rose from 101 to 108. Restoring the full-precision epilogue recovered
  101 row-rounds and removed the regression. The final merge retains that epilogue.
- Extending the A4 GDN-output path down to one token failed the independent numerical oracle.
  Compact GDN-output batches retain their qualified width-invariant A16 profile.

### Matched two-request MTP3 batches

Each request generates exactly 128 tokens with greedy sampling, KVarN K4V2-G128 KV, MTP3,
optimized proposal head and CUDA Graphs. Per-request capacity is 65,536 tokens, shared KV capacity
131,072, and prefill chunks are 2,048 tokens. Prefix reuse is disabled. Inputs are slices of
`bench/fixtures/bench_corpus.ids`, starting at offsets 0 and 128. Both requests are prepared before
timing, submitted together, and waited to completion. Each point has one discarded warmup and
three measured batches; the table reports mean batch wall time. GPU runs are serial and do not
overlap compilation.

| Prompt tokens per request | Baseline batch (s) | Merged batch (s) | Throughput ratio |
|---:|---:|---:|---:|
| 512 | 1.79937 | 1.77996 | 1.011× |
| 2,048 | 1.80622 | 1.72769 | 1.045× |
| 8,192 | 4.32420 | 4.05283 | 1.067× |
| 32,768 | 19.38697 | 18.29883 | 1.059× |

No regression was observed in these final batch measurements. This is a fixed-output-length
Engine comparison, not a claim about every prompt, model, concurrency, or maximum context.
The local paired driver and CSV reports are in `profiles/bench/upstream-9e163eee/`.

### Single-request checks

With the same KV/speculation/chunk settings, an 8,192-token capacity, one warmup and three measured
repetitions, public `ninfer_bench` reported:

| Measurement | Baseline | Merged |
|---|---:|---:|
| pp512 (tok/s) | 5,411.8 | 5,594.5 |
| pp2048 (tok/s) | 5,606.6 | 6,139.5 |
| tg128, one-token seed (tok/s) | 76.34 | 76.27 |
| pp2048 + tg128 total request (s) | 1.29079 | 1.26357 |

The short-seed decode difference is below 0.1%; prefill-only 2K samples had substantial run-to-run
variation, so the matched batch results above are the stronger end-to-end comparison. Reproduce
the single-request points with:

```bash
build/bench/ninfer_bench --weights /path/to/qwen3_8_27b_quasar.v3.ninfer \
  --corpus bench/fixtures/bench_corpus.ids -p 512,2048 -n 128 -pg '2048,128' \
  -r 3 --warmup 1 --max-ctx 8192 --prefill-chunk 2048 --kv-dtype kvarn \
  --spec mtp --draft-tokens 3 --lm-head-draft -o json --output-file /path/to/report.json
```

## RTX 5090 Laptop synchronization validation

The September 6, 2026 upstream synchronization through `ce7dee50` was measured against
`origin/master` at `0f54b98e`, built in an isolated baseline worktree before testing the merge.
These are historical public Engine micro-workloads, not the HTTP serving corpus linked below. Hardware was
an RTX 5090 Laptop GPU (24 GB), driver 595.84, CUDA 13.1.115, GCC 15.3, Release/sm_120a.
The existing Qwen3.8 QUASAR NVFP4 artifact was used unchanged for baseline and merged ordinary/MTP
runs. DFlash2 used the converted QUASAR artifact plus the pinned z-lab DFlash2 suffix. All 1,268
base object payloads compared byte-for-byte equal between the two artifacts.

Each point used one active request, CUDA Graphs, greedy sampling, a 1,024-token prefill chunk,
the checked-in token corpus, one discarded warmup, and five measured repetitions. Throughput is
the arithmetic mean in tokens/second. `tg128` has an untimed one-token seed; combined decode follows
a 2,048-token prefill and generates 128 measured decode tokens.

| KV | Backend | Baseline pp512 | Merged pp512 | Baseline tg128 | Merged tg128 | Baseline combined decode | Merged combined decode |
|---|---|---:|---:|---:|---:|---:|---:|
| INT8 | none | 4967.6 | 5669.6 | 42.2 | 42.3 | 41.9 | 42.1 |
| INT8 | MTP3 | 4678.4 | 5517.3 | 73.9 | 68.6 | 124.1 | 128.8 |
| BF16 | none | 4975.8 | 5609.9 | 42.1 | 42.2 | 41.5 | 41.7 |
| BF16 | MTP3 | 4745.8 | 5514.7 | 62.1 | 70.2 | 126.9 | 124.4 |
| INT8 | DFlash2 K15 | n/a | 5186.5 | n/a | 45.8 | n/a | 229.1 |
| BF16 | DFlash2 K15 | n/a | 5216.7 | n/a | 47.8 | n/a | 234.6 |

The short INT8 MTP3 workload regressed 7.2%; accepted proposals changed from 68 to 63 per run
and verification rounds from 60 to 64. The synchronization imports qualified numerical changes,
including FP16 V/PV and fused norm/control, so acceptance paths and greedy sequences need not
match the older implementation. This is not an across-the-board speedup: DFlash2 is substantially
faster on the longer prompt here but slower than MTP3 on the one-token seed. These short workloads
do not establish long-context, concurrent, or general-purpose serving throughput.

The merged measurement command is:

```bash
./build/bench/ninfer_bench --weights /path/to/quasar/qwen3_8_27b_nvfp4.ninfer \
  -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1 \
  --prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft \
  -o json --output-file /path/to/report.json
```

Omit speculative flags for ordinary decode; select `--spec dflash2 --draft-tokens 15` with the
DFlash2 artifact for that route, and substitute `--kv-dtype bf16` for the other KV profile.
The baseline uses its then-current `--mtp-draft-tokens 0|3` spelling.

Correctness validation included the affected independent Op oracles, real-artifact binding,
and exact ordinary-versus-MTP greedy parity for BF16/INT8, K=1..5 and concurrency 1..8.
The pre-existing baseline GDN control oracle failure at T=1024 passes with upstream's
device-residency-bounded cooperative launch. The merge also preserves canonical-column BF16
attention and single-column INT8 MTP attention, uses one fused norm/control reduction geometry,
and keeps the W8 vocabulary reduction consistent through all 48 compact MTP columns.

The converted QUASAR DFlash2 real-Engine checks passed K15/INT8/graphs/C2,
K15/BF16/eager/C8/full proposal head, K7/BF16/graphs/C3 with image/video and state restore,
and K1/INT8/graphs/C1. They check ordinary greedy agreement, nonzero accepted proposals, ragged
budgets, reproducible sampling, prefix reuse, partial terminal settlement, page boundaries,
and (at K15) ring wrap and context exhaustion. They are correctness checks, not concurrency
performance measurements.

## KVarN Long-Context Integration

`--kv-dtype kvarn` now selects K4V2-G128 with unquantized current-step attention and
committed-group encoding. Its mathematical contract and qualification commands are in
[Paged KV cache](maintainer/paged-kv-cache.md#kvarn-record-and-tail-semantics).
The previous G64 throughput, acceptance, and ordinary/speculative parity measurements do not
qualify this profile. No G128-versus-INT8 end-to-end speedup, reasoning score, or retrieval-quality
result is published here. The synchronization measurements above compare two implementations using
the same G128 codec. Benchmark matched artifacts, chunk sizes, and sampling settings before
comparing throughput; always report speculative acceptance alongside decode speed.

## Desktop Serving Results

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded historical measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

## Published coverage

Each cell links to the relevant result section. “Not published” describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | Not published |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. The Qwen3.8 NVFP4 saturation reports retain configuration and
values but no tested Git revision; the model page records that provenance limitation.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 DFlash2 outcomes](performance/qwen3.8-27b.md#dflash2-completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts), with a [README summary](../README.md#evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
