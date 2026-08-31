import Foundation
import MLX

/// Validates the native MLX runtime against Qwen's unquantized ASR checkpoint.
///
/// This deliberately does not claim to transcribe audio. It is the first gate
/// before an end-to-end runner: load the canonical BF16 tensors on the Apple
/// GPU and execute the large-vocabulary decoder operation that every generated
/// token must perform.
@main
struct MLXQwen3ASRProbe {
    private static let defaultModelDirectory = URL(
        fileURLWithPath: "/Users/pranavhari/Library/Caches/muesli-stt-benchmark/models/qwen3-asr-0.6b",
        isDirectory: true
    )

    private struct Result: Encodable {
        let experiment: String
        let model: String
        let precision: String
        let device: String
        let tensors: Int
        let decoderLayers: Int
        let lmHeadShape: [Int]
        let checkpointMapSeconds: Double
        let lmHeadArgmaxMilliseconds: Double
        let argmaxTokenID: Int32
        let note: String
    }

    static func main() {
        do {
            let modelDirectory = try parseModelDirectory()
            let weightsURL = modelDirectory.appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: weightsURL.path) else {
                throw ProbeError.missingWeights(weightsURL)
            }

            let loadStarted = ContinuousClock.now
            // MLX's safetensors loader maps on CPU; it intentionally does not
            // implement direct GPU loading. Weight movement is therefore lazy
            // and is included in the first real decoder operation below.
            let weights = try loadArrays(url: weightsURL)
            guard let lmHead = weights["thinker.lm_head.weight"] else {
                throw ProbeError.missingTensor("thinker.lm_head.weight")
            }
            guard lmHead.shape == [151_936, 1_024] else {
                throw ProbeError.unexpectedShape("thinker.lm_head.weight", lmHead.shape)
            }
            let requiredDecoderWeights = (0..<28).map { "thinker.model.layers.\($0).self_attn.q_proj.weight" }
            guard requiredDecoderWeights.allSatisfy({ weights[$0] != nil }) else {
                throw ProbeError.incompleteDecoder
            }
            let checkpointMapSeconds = seconds(loadStarted.duration(to: .now))

            let hidden = MLXArray.zeros([1, 1_024], dtype: .bfloat16, stream: .gpu)
            let decodeStarted = ContinuousClock.now
            let token = matmul(hidden, lmHead.transposed(stream: .gpu), stream: .gpu)
                .argMax(axis: -1, stream: .gpu)
            eval(token)
            let decodeMilliseconds = seconds(decodeStarted.duration(to: .now)) * 1_000

            let result = Result(
                experiment: "mlx-qwen3-asr-preflight",
                model: "Qwen/Qwen3-ASR-0.6B",
                precision: "BF16 (canonical safetensors)",
                device: "MLX GPU / Metal",
                tensors: weights.count,
                decoderLayers: 28,
                lmHeadShape: lmHead.shape,
                checkpointMapSeconds: checkpointMapSeconds,
                lmHeadArgmaxMilliseconds: decodeMilliseconds,
                argmaxTokenID: token.item(Int32.self),
                note: "Preflight only: this maps canonical weights, then measures the first GPU LM-head/argmax evaluation (including lazy movement). It is not an ASR WER or RTF result."
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(result), as: UTF8.self))
        } catch {
            fputs("mlx-qwen3-asr-probe: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    private static func parseModelDirectory() throws -> URL {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard let index = arguments.firstIndex(of: "--model-dir") else {
            return defaultModelDirectory
        }
        let valueIndex = arguments.index(after: index)
        guard valueIndex < arguments.endIndex else {
            throw ProbeError.missingModelDirectoryArgument
        }
        return URL(fileURLWithPath: arguments[valueIndex], isDirectory: true)
    }

    private static func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

private enum ProbeError: LocalizedError {
    case missingWeights(URL)
    case missingTensor(String)
    case unexpectedShape(String, [Int])
    case incompleteDecoder
    case missingModelDirectoryArgument

    var errorDescription: String? {
        switch self {
        case .missingWeights(let url):
            return "missing canonical Qwen checkpoint at \(url.path); run `hf download Qwen/Qwen3-ASR-0.6B --local-dir \(url.deletingLastPathComponent().path)` first"
        case .missingTensor(let name):
            return "canonical Qwen checkpoint is missing \(name)"
        case .unexpectedShape(let name, let shape):
            return "canonical Qwen checkpoint has unexpected \(name) shape \(shape)"
        case .incompleteDecoder:
            return "canonical Qwen checkpoint does not contain every one of the 28 decoder layers"
        case .missingModelDirectoryArgument:
            return "--model-dir requires a directory path"
        }
    }
}
