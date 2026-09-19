import Foundation
import Metal
import simd

/// Writes the tracker's 128x128 mask logits into a texture, optionally
/// applying a sigmoid, entirely on the GPU. The texture can be any size: the
/// logits are resampled bilinearly (half-pixel centers, matching the official
/// resize), which is the identity at 128x128.
public final class EfficientTAMMaskProjector
{
    public static let maskSize = 128

    private struct Uniforms
    {
        var modelSize: simd_uint2
        var outputSize: simd_uint2
        var applySigmoid: UInt32
    }

    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws
    {
        self.pipeline = try EfficientTAMComputeLibrary.pipeline(device: device, function: "efficientTAMWriteMask")
    }

    /// Encodes onto `commandBuffer` without committing it. `outputTexture`
    /// needs shader-write usage; all three color channels get the value and
    /// alpha is 1.
    public func encode(
        maskLogitsBuffer: MTLBuffer,
        outputTexture: MTLTexture,
        applySigmoid: Bool,
        commandBuffer: MTLCommandBuffer
    ) throws
    {
        let modelBytes = Self.maskSize * Self.maskSize * MemoryLayout<Float>.stride
        guard maskLogitsBuffer.length >= modelBytes else
        {
            throw EfficientTAMError("The mask-logits buffer is too small.")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            throw EfficientTAMError("Could not create the EfficientTAM mask projection encoder.")
        }
        var uniforms = Uniforms(
            modelSize: simd_uint2(UInt32(Self.maskSize), UInt32(Self.maskSize)),
            outputSize: simd_uint2(UInt32(outputTexture.width), UInt32(outputTexture.height)),
            applySigmoid: applySigmoid ? 1 : 0
        )
        encoder.label = "EfficientTAM Mask Projection"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(maskLogitsBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setTexture(outputTexture, index: 0)
        EfficientTAMComputeLibrary.dispatch(
            encoder,
            pipeline: self.pipeline,
            width: outputTexture.width,
            height: outputTexture.height
        )
        encoder.endEncoding()
    }
}
