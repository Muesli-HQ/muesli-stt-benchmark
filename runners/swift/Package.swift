// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MuesliSTTRunners",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "whisper-coreml-runner", targets: ["WhisperCoreMLRunner"])],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "WhisperCoreMLRunner",
            dependencies: [.product(name: "WhisperKit", package: "WhisperKit")],
            path: "Sources/WhisperCoreMLRunner"
        ),
    ]
)
