# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization and Python-writer/C++-reader interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- root C++ tests — core storage, runtime admission/resource policy, public API, serving protocols,
  logging, benchmark reports and causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, `models/qwen3_5/`
and `ops/`. Registration helpers live in `cmake/NinferTests.cmake`; included manifests keep
executables and CTest working directories under `build/tests/`.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/Q8, FP8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Alternatively, `cmake --preset dev` enables products, tests and benchmarks together.
After building, `ctest --preset dev` runs the same CTest suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

On CPU-only machines, build the native renderer without the CUDA/media dependencies before running
the Python suite (with `tests/requirements-cpu.txt` and CPU PyTorch installed):

```bash
cmake -S tests/text -B build/cpu-text -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-text -j
ctest --test-dir build/cpu-text --output-on-failure
NINFER_JINJA_TEST="$PWD/build/cpu-text/ninfer_jinja_test" python3.11 -m pytest tests
PYTHONPATH=eval python3.11 -m unittest discover -s eval/tests -p 'test_*.py'
```

`NINFER_JINJA_TEST` selects the renderer for pytest. The regular CTest registration passes its built
renderer explicitly; a missing renderer is an error, so the independent C++/Python comparison is
also exercised by CPU CI.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

The variable-width DFlash2 target-attention subset can be run with
`./build/tests/ninfer_softmax_attention_test --dflash2-only`. It covers D256/Q24/KV4 across all five
cache codecs, W=2..16, B=1..8, request-local prefixes, cache effects, and Graph metadata/input
updates. The default executable also runs the existing attention geometries and prefill tests.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_q8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

The Linear, LinearAdd and LinearSwiGLU common `.cpp` implementations each compile once into a
test support library. Both those libraries and the Op test executables receive the oracle's
`-fno-fast-math` and `-ffp-contract=off` options on GNU/Clang C++ compilers.

Run all self-contained CPU tests with Python 3.11 and the dependencies used by CI:

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"
python3 -m pip install -r tests/requirements-cpu.txt
python3 -m pytest tests
PYTHONPATH=eval python3 -m unittest discover -s eval/tests -p 'test_*.py'
```

The Python suites exercise conversion and encoded output, without running model inference.
The maintained environment uses Python 3.11 with the dependencies for those suites. C++ binding and Engine tests
cover consumption of their resulting representation.

The real loading test accepts an explicit artifact path and optional component selection:

```bash
./build/tests/ninfer_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_score_real_test --output-on-failure
```

The fork's greedy-MTP regression compares committed token IDs against ordinary decode for
BF16, INT8, FP8, NVFP4, K8V4 and KVarN KV, draft counts 1..15, and selected concurrency 1..8:

```bash
NINFER_TEST_ARTIFACT=/path/to/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_mtp_greedy_parity_real_test --output-on-failure
```

For a focused reproduction, use `--draft-tokens 4 --concurrency 7 --kv-dtype bf16`.
Without selectors it runs all six formats at fixed depths 1..5 and 15 at concurrency 1, using the thinking-code
fixture and 512 output tokens. Each case runs twice and compares against ordinary greedy output.
`--kv-dtype kvarn --sample 1 --output-tokens 8192 --draft-tokens 3` checks long KVarN generation
against ordinary decoding through the 4K/8K arithmetic transitions. Use `--kv-dtype int8` for the
corresponding INT8 gate. Selecting a draft depth always retains the no-MTP reference run.
An explicit `--corpus PATH` enables long resident prompts: samples 3..7 contain 8,190, 32,799,
122,879, 196,607, and 245,743 tokens. Use `--prefix-reuse --concurrency 2` to exercise restored,
unequal concurrent rows, or `--no-cuda-graph --full-proposal-head` for eager/full-head execution.
Sample 8 uses eight distinct 231-token corpus slices (offsets `128 * row`) and equal output
budgets to exercise short, low-acceptance compact batches. The C8/MTP3 regression command is:

```bash
NINFER_TEST_ARTIFACT=/path/to/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_mtp_greedy_parity_real_test \
  --corpus bench/fixtures/bench_corpus.ids --sample 8 --concurrency 8 \
  --kv-dtype kvarn --draft-tokens 3 --prefill-chunk 2048 --output-tokens 513
```

`--adaptive --draft-tokens 15` compares adaptive windows against MTP-off, retaining the same KV
algorithm. `--tool-loop --kv-dtype kvarn` checks three greedy turns with 27 tool schemas and long
tool results, including endpoint reuse and response replay after client-rewritten reasoning.
It compares token IDs, reasoning/content, finish reason, and tool arguments against MTP-off.
The same KVarN harness accepts `--spec dflash2`, with draft counts 1, 3, 7, and 15, against the
same-backend repeatability criterion; its artifact must contain the DFlash2 companion.
KVarN MTP verification stops at group-publication boundaries so later queries see the same
compressed history as ordinary decoding. FP8/NVFP4/K8V4 narrow attention launches independent
canonical query columns together, with width- and batch-independent split schedules.
The Op suite independently checks raw-current-chunk attention, represented history, and accepted
group encoding against mathematical oracles. It also checks exact ordinary/MTP attention-column
parity for masked, mixed-length batches, including split-policy transitions and CUDA Graph replay.

The DFlash2 test also accepts the converted QUASAR artifact. Positional arguments select draft
count, graphs, optimized proposal head, concurrency, KV storage, and optional Vision:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_quasar.ninfer \
  ./build/tests/ninfer_qwen3_5_dflash2_real_test 15 1 1 2 int8
```

KV storage can be `bf16`, `int8`, or `kvarn`. Append `1 0` to enable Vision and force Host
StateImage snapshots with zero extra Device slots. The image/video reuse checks require zero
Vision execution on the reused request and new Vision execution for an appended media item.
Same-input reuse checks exact generated-token agreement; new-media and KVarN visual-bridge checks
compare identical request schedules across Host/Device checkpoint placement, not against a
numerically different full-prefill schedule. The MTP KVarN Vision continuation suite is selected by
`NINFER_PREFIX_REAL_SCENARIO=kvarn-vision` with `NINFER_TEST_ARTIFACT` set on
`ninfer_qwen3_5_prefix_real_test`. It runs MTP5 with four and zero extra Device slots, checks
stop/resume at the 255-token committed frontier, exercises cross-chunk Vision and media follow-ups,
and requires actual Host StateImage capture and restoration in the zero-slot configuration.

Run the 35B-A3B MoE route independently:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_moe_real_test --output-on-failure
```

Without `NINFER_TEST_ARTIFACT`, CTest marks these real Engine tests as skipped. Run GPU integration
tests serially. `NINFER_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as `vision`,
`pressure-resume` or `concurrent`; the default is `all`. These integration checks
use behavior and state accounting rather than another numerical path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares stripped and preserved closed-turn prompt lengths, and verifies compatible
prefix reuse, speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## DFlash2 Engine integration

The real test uses an artifact containing DFlash2 and checks output budgets, speculative activity,
penalty-enabled sampling, compact batches with unequal budgets, same-route same-seed replay,
retained/fresh prefix behavior and absence of a full backend KV pool. A shared DFlash/DFlash2 fixture starts decode at token 63, verifies across the page
boundary, stops after one target column at token 64, and checks the exact retained frontier and
subsequent generation with and without reuse.
The KV Store test checks exact mapping and reservation accounting for the same transition.
K>=7 also exercises a stop inside a licensed block; K=15 additionally checks oversized prefill,
local ring wrap, and the logical context-capacity tail. Optional Vision runs image/video capture
and prefix restore. Zero extra Device StateImage slots exercise Host snapshot/restore.

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash2_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 15 1 1 8
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16` or `int8`),
Vision enabled, and extra Device StateImage slots. Defaults are `15 1 1 8 bf16 0 3`. Run GPU
integration tests serially. The individual Op suites remain the numerical/state-transition oracle;
the fixed Engine fixture does not define bit parity across arbitrary floating-point routes.
