// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MPS-EfficientTAM",
    platforms: [.macOS("15.0"), .iOS("18.0"), .visionOS("2.0")],
    products: [
        .library(name: "MPSEfficientTAM", targets: ["MPSEfficientTAM"]),
    ],
    targets: [
        .target(
            name: "MPSEfficientTAM",
            resources: [
                .copy("Models"),
                .copy("Utils/Compute"),
            ]
        ),
        .testTarget(
            name: "MPSEfficientTAMTests",
            dependencies: ["MPSEfficientTAM"],
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageVersions: [.v5]
)
