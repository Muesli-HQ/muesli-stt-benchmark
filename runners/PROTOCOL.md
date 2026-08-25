# Persistent runtime-runner protocol

A runner owns one loaded model for the life of its process. The controller starts a
fresh process for each model; the first `transcribe` request is cold and subsequent
requests are warm. Communication is JSON Lines over standard input/output. Logs go
to standard error only.

Controller requests:

    {"type":"initialize","schema_version":1,"model":"whisper-tiny","runtime":"coreml"}
    {"type":"transcribe","id":"clip-1","audio":"/absolute/clip.wav","pass":0,"reference":"..."}
    {"type":"shutdown"}

Runner responses:

    {"type":"ready","schema_version":1}
    {"type":"result","transcript":"recognized text","runtime_metadata":{"execution_units":"cpuAndNeuralEngine"}}
    {"type":"error","error":"audio is longer than this model's supported window"}

A runner must not download model weights during initialization. It should return an
error response for an unavailable model or an unsupported individual clip without
ending the process.

## Runtime ownership

| Runtime | Workload | Required metadata |
| --- | --- | --- |
| Core ML | ASR | Model path, compute units, and later Instruments placement evidence. |
| LiteRT | Multimodal ASR or cleanup | Metal/CPU/audio executor configuration and model cache. |
| GGUF | Text cleanup | llama.cpp settings and model quantization. |
| MLX | Text cleanup or decoder experiments | MLX version and Metal settings. |
| Apple Speech | ASR | Recognizer locale and on-device requirement. |

GGUF and MLX cleanup use a transcript hypothesis as input rather than audio. Their
quality must be reported separately from direct audio-to-text ASR.
