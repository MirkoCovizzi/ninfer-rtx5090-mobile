# Single-GPU serving performance

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
