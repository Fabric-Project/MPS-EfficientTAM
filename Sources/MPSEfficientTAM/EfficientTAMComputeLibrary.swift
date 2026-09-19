import Foundation
import Metal

/// Loads the package's Metal source from its bundle resources and builds
/// compute pipelines from it.
enum EfficientTAMComputeLibrary
{
    static func pipeline(device: MTLDevice, function name: String) throws -> MTLComputePipelineState
    {
        guard
            let shaderURL = Bundle.module.url(
                forResource: "EfficientTAMCompute",
                withExtension: "metal",
                subdirectory: "Compute"
            ),
            let source = try? String(contentsOf: shaderURL, encoding: .utf8),
            let library = try? device.makeLibrary(source: source, options: nil),
            let function = library.makeFunction(name: name)
        else
        {
            throw EfficientTAMError("Could not load the EfficientTAM Metal kernel \(name).")
        }
        return try device.makeComputePipelineState(function: function)
    }

    static func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        width: Int,
        height: Int
    )
    {
        let threadWidth = pipeline.threadExecutionWidth
        let threadHeight = max(1, pipeline.maxTotalThreadsPerThreadgroup / threadWidth)
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        )
    }
}
