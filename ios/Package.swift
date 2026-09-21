// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ElderHelpCore",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "ElderHelpCore", targets: ["ElderHelpCore"])],
    targets: [
        .target(name: "ElderHelpCore", path: "ElderHelp/Core"),
        .testTarget(name: "ElderHelpCoreTests", dependencies: ["ElderHelpCore"], path: "ElderHelpTests", resources: [.copy("Fixtures")])
    ]
)
