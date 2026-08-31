# Qwen3-ASR-0.6B on MLX: baseline plan

This is an experimental runtime investigation. It does not change the Muesli
app and it must not produce a published ASR comparison until the full path is
implemented and verified.

## Current integration seam

The benchmark controller (`stt-benchmark session`) owns workload iteration,
the one-cold-plus-three-warm policy, wall time, RTF, and normalized WER. A
runtime supplies a persistent JSONL runner. The existing Core ML Whisper and
LiteRT Gemma runners establish the format: initialize once, keep models loaded,
serve each audio request, then shut down.

In the Muesli app, the corresponding Qwen route is deliberately separate from
the benchmark repository:

```text
TranscriptionRuntime
  -> Qwen3AsrTranscriber
     -> MuesliQwen3AsrManager
        -> mel frontend -> Core ML audio encoder
        -> prompt/audio embedding merge -> stateful Core ML decoder
```

The smallest clean benchmark integration is therefore a new persistent
`mlx-qwen3-asr-runner` beside the existing standalone runners. It will use the
same controller and manifest, without depending on `muesli-cli` or changing
product code.

## Canonical model facts

The downloaded `Qwen/Qwen3-ASR-0.6B` checkpoint is 1.88 GB and contains 612
BF16 safetensors. Its config and tensor keys verify:

| Component | Canonical configuration |
|---|---|
| Audio frontend | Whisper feature extractor, 16 kHz, 128 mel bins, 400 FFT, 160 hop |
| Audio encoder | 18 layers, model dim 896, 14 heads, output dim 1024 |
| Text decoder | 28 layers, hidden 1024, 16 query heads, 8 KV heads, head dim 128 |
| MLP / norms | SwiGLU (3072 intermediate), RMSNorm epsilon 1e-6, Q/K RMSNorm |
| Positional encoding | Qwen M-RoPE, theta 1,000,000, interleaved sections [24, 20, 20] |
| Vocabulary | 151,936 tokens |

The installed FluidAudio/Muesli artifact is an **int8 compiled Core ML** encoder
and decoder plus embedding data. It is suitable as a comparison baseline and
for a later hybrid experiment, but it cannot be used as the source of the
canonical MLX BF16 weights.

## Phase 2 completion: native MLX preflight

`runners/swift/Sources/MLXQwen3ASRProbe` uses official `mlx-swift` 0.31.6 and
the XcodeGen project in `runners/xcode`. It validates all 28 decoder layers,
maps the original safetensors, and evaluates the actual BF16
`[1, 1024] × [1024, 151936] -> argmax` operation on MLX Metal.

The initial local run completed successfully. Its approximately 1.013 s
measurement includes lazy transfer of the 311 MB LM head; it is a runtime
bring-up observation, not a decoded-token timing and not an ASR result.

MLX's safetensors loader maps weights through the CPU; direct `stream: .gpu`
loading is currently unimplemented. The first real GPU operation consequently
includes lazy movement of the tensors it uses. The eventual runner must expose
that separately as cold model load / first-token work, then measure warm tokens
after the relevant weights are resident.

## End-to-end baseline gates

Before calling a result an MLX Qwen ASR benchmark, implement these in order:

1. Exact 16 kHz audio preprocessing and the 18-layer BF16 audio encoder. Check
   encoder output against the reference model on fixed clips before measuring.
2. Exact prompt/audio embedding merge, tokenizer decoding, Q/K RMSNorm,
   interleaved M-RoPE, GQA, and greedy stop-token behavior in the 28-layer
   decoder. Check generated token IDs against the reference on fixed encoder
   features.
3. A persistent JSONL `mlx-qwen3-asr-runner` that returns real transcripts to
   `stt-benchmark session`. It must report preprocessing, encoder, projection,
   decoder prefill, decode total, generated-token count, and p50/p95 token time.
4. Run the existing local clip manifest as one cold pass followed by three warm
   passes, then publish WER/RTF only after every attempt is successful.

The initial full baseline should use ordinary MLX operations and BF16 weights.
`MLXFast.scaledDotProductAttention` and `MLXFast.rmsNorm` are candidates only
after the unoptimized path matches output. Compiled graphs, a preallocated KV
cache, precision changes, and custom Metal kernels are separate experiments;
one change per measured commit.

## Comparison policy

A future hybrid Core ML encoder + MLX decoder is useful to isolate the
autoregressive decoder, but it is a distinct runtime placement and must be
reported as such. It cannot be presented as an all-MLX result, nor compared
directly with a full-BF16 MLX baseline without naming the encoder precision and
placement.
