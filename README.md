# Muesli STT Benchmark

Reproducible, machine-aware benchmarks for on-device automatic speech
recognition. The project compares model quality, latency, throughput, memory,
and energy across real Apple hardware and runtimes such as Core ML/ANE,
LiteRT-LM/Metal, GGUF/llama.cpp, and MLX.

It is intentionally separate from the Muesli app: an experiment can add a
model adapter or workload without changing product code, and a published result
always names the exact machine and runtime that produced it.

## What a benchmark records

- model, revision, quantization, runtime, and runtime settings;
- machine profile: macOS, chip, unified memory, power state where available;
- fixed audio workload and duration bucket;
- transcript quality: corpus WER plus substitutions, deletions, and insertions;
- performance: per-clip wall time, real-time factor, first-result latency when
  supplied by an adapter, and failures;
- optional Instruments/energy artifacts, stored outside Git and referenced by
  path or published separately.

An ANE result is only called ANE-backed after a Core ML Instruments trace has
verified placement. Declaring `computeUnits = .all` is not sufficient evidence.

## Initial workloads

| Workload | Source | Purpose |
|---|---|---|
| `librispeech-clean` | `openslr/librispeech_asr`, `clean/test` | Clean read-English quality baseline |
| `earnings22` | `distil-whisper/earnings22`, `chunked/test` | Compressed, accented, disfluent real-world speech |
| local manifests | supplied later | Dictation, meeting, noisy, and duration-specific clips |

The downloader writes a manifest of normalized 16 kHz mono PCM WAVs. Audio and
weights are deliberately ignored by Git; check each upstream dataset's license
before redistribution.

## Quick start

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .

# Pull a reproducible small public sample (default: 40 clips per workload).
./.venv/bin/stt-benchmark fetch --count 40

# Capture the machine on which a run will occur.
./.venv/bin/stt-benchmark machine --output results/machine-m5.json

# Benchmark a Muesli CLI model through the generic command adapter.
./.venv/bin/stt-benchmark run \
  --manifest data/librispeech-clean/refs.jsonl \
  --adapter configs/adapters/muesli-cli.json \
  --model parakeet-unified --runtime coreml \
  --machine results/machine-m5.json \
  --output results/librispeech-clean--parakeet-unified--coreml.jsonl

./.venv/bin/stt-benchmark report results/librispeech-clean--parakeet-unified--coreml.jsonl
```

`run` accepts an adapter JSON file whose command uses `{audio}` and `{model}`.
The bundled adapter parses Muesli CLI JSON. Other runtimes use the same output
contract by adding an adapter rather than modifying the runner.

## Duration and long-audio policy

Every input records its `duration_seconds`; reports group results by explicit
duration buckets. Start with `2–10s`, `10–30s`, `30–60s`, and `60–300s` so
startup costs, encoder scaling, and long-context behavior are visible instead
of being hidden in one average. Streaming runs should additionally publish
partial/final latency and stability metrics.

For models with fixed audio windows, the adapter must declare its chunk and
overlap policy. A model must not be described as handling long audio merely
because the harness concatenates chunk transcripts.

## Adding a runtime adapter

1. Add `configs/adapters/<runtime>-<model>.json` with a versioned command.
2. Make it emit a transcript in the configured result format.
3. Document model revision, quantization, execution units, chunking, and
   decoding parameters in the run invocation or a checked-in experiment note.
4. Run at least one public workload and publish its JSONL plus machine profile.

Core ML, LiteRT, GGUF, and MLX are benchmark dimensions, not interchangeable
labels: record the actual backend and relevant placement, such as ANE encoder
plus Metal decoder.

## Experimental MLX: Qwen3-ASR-0.6B

`Qwen/Qwen3-ASR-0.6B` is the first MLX experiment because it has a substantial
autoregressive decoder: an 18-layer audio encoder feeds a 28-layer Qwen3 GQA
decoder. The model's canonical checkpoint is BF16; the first end-to-end
baseline will retain that precision and use neither quantization nor custom
Metal kernels.

The native runner must be built with Xcode, not `swift run`: MLX's Metal
library is packaged by the Xcode product. One-time host setup is:

```bash
xcodebuild -downloadComponent MetalToolchain
hf download Qwen/Qwen3-ASR-0.6B \
  --local-dir /Users/you/Library/Caches/muesli-stt-benchmark/models/qwen3-asr-0.6b

xcodegen generate --spec runners/xcode/project.yml --project-root runners/xcode
xcodebuild -skipPackagePluginValidation \
  -project runners/xcode/MLXQwen3ASRProbe.xcodeproj \
  -scheme MLXQwen3ASRProbe -configuration Debug \
  -derivedDataPath /Users/you/Library/Caches/muesli-spm/stt-benchmark/mlx-xcode build
```

Run the resulting `mlx-qwen3-asr-probe` binary to validate that MLX can map the
canonical safetensors and execute the real BF16 decoder LM-head/argmax on Metal.
This probe is intentionally **not** an ASR benchmark: it has no audio frontend,
no transcript, and therefore no WER or RTF result. The exact implementation
gates for the end-to-end runner are in [the MLX experiment note](docs/mlx-qwen3-asr.md).

## Repository layout

```text
configs/adapters/  command adapters for products and runtimes
datasets/          public workload policy and dataset documentation
src/               fetch, machine-profile, execution, and reporting tooling
results/           ignored local results (publish curated result files separately)
runners/xcode/     XcodeGen project for MLX executables that need a metallib
```

## Provenance

This starts from Muesli's earlier STT harness, which used LibriSpeech-clean and
Earnings-22 and drove `muesli-cli` to calculate WER. The app repository retained
the CLI integration but intentionally did not merge the Python evaluation
tooling; this repository is the independent home for that work.
