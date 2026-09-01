import Foundation
import Darwin
import Qwen3ASR
import Qwen3Common

/// JSONL adapter for the Apache-2.0 Qwen3-ASR MLX Swift implementation.
///
/// This runner deliberately uses the MLX-community 4-bit conversion. Its
/// responses identify that precision so it cannot be mistaken for the planned
/// canonical BF16 runtime baseline. Model construction is deferred to the
/// first request: pass 0 therefore includes weight mapping and GPU warm-up.
@main
@MainActor
struct MLXQwen3ASRRunner {
    private static let modelID = "mlx-community/Qwen3-ASR-0.6B-4bit"
    private static var model: Qwen3ASRModel?
    private static let protocolOutput = dup(STDOUT_FILENO)

    static func main() async {
        // The third-party model library uses print() while loading. Preserve a
        // strict JSONL protocol by sending those diagnostic lines to stderr;
        // structured responses continue through the saved stdout descriptor.
        _ = protocolOutput
        dup2(STDERR_FILENO, STDOUT_FILENO)
        while let line = readLine() {
            do {
                let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
                guard let type = request?["type"] as? String else {
                    throw RunnerError.invalidRequest("missing request type")
                }
                switch type {
                case "initialize":
                    try write(["type": "ready", "schema_version": 1])
                case "transcribe":
                    guard let audioPath = request?["audio"] as? String else {
                        throw RunnerError.invalidRequest("missing audio path")
                    }
                    let started = ContinuousClock.now
                    let didLoadModel = model == nil
                    if model == nil {
                        model = try await Qwen3ASRModel.fromPretrained(modelId: modelID)
                    }
                    let loadSeconds = seconds(started.duration(to: .now))
                    guard let model else { throw RunnerError.modelUnavailable }

                    let audioStarted = ContinuousClock.now
                    let samples = try AudioFileLoader.load(url: URL(fileURLWithPath: audioPath), targetSampleRate: 16_000)
                    let preprocessingSeconds = seconds(audioStarted.duration(to: .now))

                    let inferenceStarted = ContinuousClock.now
                    let transcript = model.transcribe(audio: samples, sampleRate: 16_000, language: "English")
                    let inferenceSeconds = seconds(inferenceStarted.duration(to: .now))
                    let totalSeconds = seconds(started.duration(to: .now))

                    try write([
                        "type": "result",
                        "transcript": transcript,
                        "runtime_metadata": [
                            "runtime": "mlx",
                            "model": modelID,
                            "precision": "MLX-community 4-bit conversion (exploratory; not canonical BF16)",
                            "device": "Apple GPU / Metal via MLX",
                            "model_load_and_warmup_seconds": didLoadModel ? loadSeconds : 0.0,
                            "audio_loading_seconds": preprocessingSeconds,
                            "inference_seconds": inferenceSeconds,
                            "runner_total_seconds": totalSeconds,
                        ],
                    ])
                case "shutdown":
                    try write(["type": "shutdown"])
                    return
                default:
                    throw RunnerError.invalidRequest("unsupported request type: \(type)")
                }
            } catch {
                fputs("mlx-qwen3-asr-runner: \(error.localizedDescription)\n", stderr)
                try? write(["type": "error", "error": error.localizedDescription])
            }
        }
    }

    private static func write(_ object: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        var bytes = Array(data)
        bytes.append(0x0A)
        try bytes.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            var written = 0
            while written < rawBuffer.count {
                let count = Darwin.write(protocolOutput, baseAddress.advanced(by: written), rawBuffer.count - written)
                guard count > 0 else {
                    throw RunnerError.protocolWriteFailed
                }
                written += count
            }
        }
    }

    private static func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

private enum RunnerError: LocalizedError {
    case invalidRequest(String)
    case modelUnavailable
    case protocolWriteFailed

    var errorDescription: String? {
        switch self {
        case .invalidRequest(let detail): return detail
        case .modelUnavailable: return "MLX Qwen3-ASR model was not initialized"
        case .protocolWriteFailed: return "could not write JSONL protocol response"
        }
    }
}
