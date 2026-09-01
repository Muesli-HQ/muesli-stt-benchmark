// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MuesliSTTRunners",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "whisper-coreml-runner", targets: ["WhisperCoreMLRunner"]),
        .executable(name: "gemma-litert-runner", targets: ["GemmaLiteRTRunner"]),
        .executable(name: "mlx-qwen3-asr-probe", targets: ["MLXQwen3ASRProbe"]),
        .executable(name: "mlx-qwen3-asr-runner", targets: ["MLXQwen3ASRRunner"]),
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        // Apache-2.0 ASR architecture used only for the MLX-native exploratory
        // runner. The benchmark labels its converted 4-bit checkpoint clearly.
        .package(
            url: "https://github.com/vfasky/qwen3-asr-swift.git",
            revision: "4824c95e1e4624200405d639fb4ebe10f93f1075"
        ),
    ],
    targets: [
        .binaryTarget(
            name: "CLiteRTLM",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.13.1/CLiteRTLM_mac.xcframework.zip",
            checksum: "ec9ffe230dc39117a7fc8933b1cc15910454027fee6d3041534ab7cf17313981"
        ),
        .executableTarget(
            name: "WhisperCoreMLRunner",
            dependencies: [.product(name: "WhisperKit", package: "WhisperKit")],
            path: "Sources/WhisperCoreMLRunner"
        ),
        .executableTarget(
            name: "GemmaLiteRTRunner",
            dependencies: ["CLiteRTLM"],
            path: "Sources/GemmaLiteRTRunner"
        ),
        .executableTarget(
            name: "MLXQwen3ASRProbe",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MLXQwen3ASRProbe"
        ),
        .executableTarget(
            name: "MLXQwen3ASRRunner",
            dependencies: [
                .product(name: "Qwen3ASR", package: "qwen3-asr-swift"),
                .product(name: "Qwen3Common", package: "qwen3-asr-swift"),
            ],
            path: "Sources/MLXQwen3ASRRunner"
        ),
    ]
)
