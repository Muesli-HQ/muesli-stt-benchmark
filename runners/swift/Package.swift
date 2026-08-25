// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MuesliSTTRunners",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "whisper-coreml-runner", targets: ["WhisperCoreMLRunner"]),
        .executable(name: "gemma-litert-runner", targets: ["GemmaLiteRTRunner"]),
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", branch: "main"),
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
    ]
)
