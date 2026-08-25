import Foundation
import WhisperKit

private struct Request: Decodable {
    let type: String
    let schemaVersion: Int?
    let model: String?
    let runtime: String?
    let id: String?
    let audio: String?
}

private struct Response: Encodable {
    let type: String
    let schemaVersion: Int?
    let transcript: String?
    let error: String?
    let runtimeMetadata: [String: String]?

    init(type: String, schemaVersion: Int? = nil, transcript: String? = nil, error: String? = nil, runtimeMetadata: [String: String]? = nil) {
        self.type = type
        self.schemaVersion = schemaVersion
        self.transcript = transcript
        self.error = error
        self.runtimeMetadata = runtimeMetadata
    }
}

@main
struct WhisperCoreMLRunner {
    static func main() async {
        var whisperKit: WhisperKit?
        var modelName: String?
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()

        func emit(_ response: Response) {
            guard let data = try? encoder.encode(response), let line = String(data: data, encoding: .utf8) else { return }
            print(line)
            fflush(stdout)
        }

        while let line = readLine() {
            guard let request = try? decoder.decode(Request.self, from: Data(line.utf8)) else {
                emit(Response(type: "error", error: "invalid JSONL request"))
                continue
            }
            switch request.type {
            case "initialize":
                guard request.runtime == "coreml", let requested = request.model, let folder = modelFolder(for: requested) else {
                    emit(Response(type: "error", error: "unsupported Whisper Core ML model or runtime"))
                    continue
                }
                guard FileManager.default.fileExists(atPath: folder.path) else {
                    emit(Response(type: "error", error: "model is not installed locally: \(folder.path)"))
                    continue
                }
                do {
                    whisperKit = try await WhisperKit(WhisperKitConfig(
                        modelFolder: folder.path,
                        computeOptions: ModelComputeOptions(
                            audioEncoderCompute: .cpuAndNeuralEngine,
                            textDecoderCompute: .cpuAndNeuralEngine
                        )
                    ))
                    modelName = requested
                    emit(Response(type: "ready", schemaVersion: 1, runtimeMetadata: [
                        "runtime": "coreml", "execution_units": "cpuAndNeuralEngine",
                        "model_folder": folder.path,
                    ]))
                } catch {
                    emit(Response(type: "error", error: error.localizedDescription))
                }
            case "transcribe":
                guard let whisperKit, let modelName, let audio = request.audio else {
                    emit(Response(type: "error", error: "runner is not initialized"))
                    continue
                }
                do {
                    let options = modelName.hasSuffix(".en") ? DecodingOptions() : DecodingOptions(detectLanguage: true)
                    let results = try await whisperKit.transcribe(audioPath: audio, decodeOptions: options)
                    let transcript = results.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
                    emit(Response(type: "result", transcript: transcript, runtimeMetadata: ["runtime": "coreml"]))
                } catch {
                    emit(Response(type: "error", error: error.localizedDescription))
                }
            case "shutdown":
                return
            default:
                emit(Response(type: "error", error: "unsupported request type: \(request.type)"))
            }
        }
    }

    private static func modelFolder(for model: String) -> URL? {
        let name: String
        switch model {
        case "whisper-tiny": name = "openai_whisper-tiny"
        case "whisper-tiny-english": name = "openai_whisper-tiny.en"
        case "whisper-large-turbo": name = "openai_whisper-large-v3-v20240930_626MB"
        default: return nil
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/huggingface/models/argmaxinc/whisperkit-coreml", isDirectory: true)
            .appendingPathComponent(name, isDirectory: true)
    }
}
