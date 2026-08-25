import AVFoundation
import CLiteRTLM
import Foundation

private struct Request: Decodable { let type: String; let model: String?; let runtime: String?; let audio: String? }
private struct Response: Encodable {
    let type: String; let schemaVersion: Int?; let transcript: String?; let error: String?; let runtimeMetadata: [String: String]?
    init(_ type: String, transcript: String? = nil, error: String? = nil, metadata: [String: String]? = nil) { self.type = type; self.schemaVersion = type == "ready" ? 1 : nil; self.transcript = transcript; self.error = error; self.runtimeMetadata = metadata }
}

@main struct GemmaLiteRTRunner {
    static func main() async {
        let encoder = JSONEncoder(); let decoder = JSONDecoder()
        var engine: OpaquePointer?
        defer { if let engine { litert_lm_engine_delete(engine) } }
        func emit(_ response: Response) { if let data = try? encoder.encode(response) { FileHandle.standardOutput.write(data); FileHandle.standardOutput.write(Data("\n".utf8)) } }
        while let line = readLine() {
            guard let request = try? decoder.decode(Request.self, from: Data(line.utf8)) else { emit(Response("error", error: "invalid JSONL request")); continue }
            switch request.type {
            case "initialize":
                guard request.runtime == "litert", let model = request.model, let modelURL = modelURL(model), FileManager.default.fileExists(atPath: modelURL.path) else { emit(Response("error", error: "Gemma LiteRT model is not installed locally")); continue }
                let cache = modelURL.deletingLastPathComponent().appendingPathComponent("litert-cache", isDirectory: true)
                try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
                guard let settings = litert_lm_engine_settings_create(modelURL.path, "gpu", nil, "cpu") else { emit(Response("error", error: "could not create LiteRT settings")); continue }
                litert_lm_engine_settings_set_max_num_tokens(settings, 4096)
                litert_lm_engine_settings_set_cache_dir(settings, cache.path)
                engine = litert_lm_engine_create(settings)
                litert_lm_engine_settings_delete(settings)
                guard engine != nil else { emit(Response("error", error: "could not create LiteRT engine")); continue }
                emit(Response("ready", metadata: ["runtime": "litert", "backend": "gpu", "audio_executor": "cpu", "model_path": modelURL.path]))
            case "transcribe":
                guard let engine, let audio = request.audio else { emit(Response("error", error: "runner is not initialized")); continue }
                let audioURL = URL(fileURLWithPath: audio)
                let duration = (try? await AVURLAsset(url: audioURL).load(.duration)).map(CMTimeGetSeconds) ?? 0
                guard duration <= 30 else { emit(Response("error", error: "Gemma supports audio clips up to 30 seconds; this clip is \(String(format: "%.1f", duration)) seconds.")); continue }
                guard let wavURL = try? prepareWAV(audioURL) else { emit(Response("error", error: "could not prepare 16 kHz mono WAV audio")); continue }
                defer { if wavURL != audioURL { try? FileManager.default.removeItem(at: wavURL) } }
                guard let session = litert_lm_session_config_create(), let conversationConfig = litert_lm_conversation_config_create(), let optional = litert_lm_conversation_optional_args_create() else { emit(Response("error", error: "could not create LiteRT conversation")); continue }
                defer { litert_lm_session_config_delete(session); litert_lm_conversation_config_delete(conversationConfig); litert_lm_conversation_optional_args_delete(optional) }
                litert_lm_session_config_set_max_output_tokens(session, 128)
                var sampler = LiteRtLmSamplerParams(type: kLiteRtLmSamplerTypeTopP, top_k: 1, top_p: 0.95, temperature: 1, seed: 0)
                litert_lm_session_config_set_sampler_params(session, &sampler)
                litert_lm_conversation_config_set_session_config(conversationConfig, session)
                guard let conversation = litert_lm_conversation_create(engine, conversationConfig) else { emit(Response("error", error: "could not create LiteRT conversation")); continue }
                defer { litert_lm_conversation_delete(conversation) }
                let prompt = "Transcribe the following speech segment in its original language. Only output the transcription, with no newlines."
                let message: [String: Any] = ["role": "user", "content": [["type": "text", "text": prompt], ["type": "audio", "path": wavURL.path]]]
                guard let data = try? JSONSerialization.data(withJSONObject: message), let json = String(data: data, encoding: .utf8), let response = litert_lm_conversation_send_message(conversation, json, nil, optional) else { emit(Response("error", error: "LiteRT transcription failed")); continue }
                defer { litert_lm_json_response_delete(response) }
                guard let value = litert_lm_json_response_get_string(response), let object = try? JSONSerialization.jsonObject(with: Data(String(cString: value).utf8)) as? [String: Any], let content = object["content"] as? [[String: Any]] else { emit(Response("error", error: "LiteRT returned an invalid response")); continue }
                let transcript = content.compactMap { $0["text"] as? String }.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
                emit(Response("result", transcript: transcript, metadata: ["runtime": "litert", "backend": "gpu", "audio_executor": "cpu"]))
            case "shutdown": return
            default: emit(Response("error", error: "unsupported request type"))
            }
        }
    }
    static func modelURL(_ model: String) -> URL? {
        let filename: String; let directory: String
        switch model { case "gemma-4-e2b": filename = "gemma-4-E2B-it.litertlm"; directory = "gemma-4-e2b-litert-lm"; case "gemma-4-e4b": filename = "gemma-4-E4B-it.litertlm"; directory = "gemma-4-e4b-litert-lm"; default: return nil }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/muesli/models/\(directory)/\(filename)")
    }
    static func prepareWAV(_ source: URL) throws -> URL {
        if source.pathExtension.lowercased() == "wav" { return source }
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
        process.arguments = ["-f", "WAVE", "-d", "LEI16@16000", "-c", "1", source.path, destination.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw NSError(domain: "GemmaLiteRTRunner", code: Int(process.terminationStatus)) }
        return destination
    }
}
