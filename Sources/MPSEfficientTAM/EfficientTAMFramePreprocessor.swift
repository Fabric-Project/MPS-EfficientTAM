import Foundation
import Metal
import simd

/// Encodes a resample of an arbitrary source texture directly into the
/// tightly-packed NHWC float32 RGB buffer `EfficientTAMImageEncoder` requires
/// (512x512, values as sampled), with no CPU-side copy.
///
/// Texels are passed through unchanged: no color-space conversion and no
/// normalization. The model applies its own ImageNet normalization, and the
/// host decides what color space the texture is in.
public final class EfficientTAMFramePreprocessor
{
    private struct Uniforms
    {
        var outputSize: simd_uint2
        var textureTransform: simd_float4x4
    }

    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws
    {
        self.pipeline = try EfficientTAMComputeLibrary.pipeline(device: device, function: "efficientTAMPrepareRGB")
    }

    /// The buffer `encode` writes into must hold at least this many bytes.
    public var outputBufferLength: Int
    {
        EfficientTAMImageEncoder.inputWidth * EfficientTAMImageEncoder.inputHeight * 3 * MemoryLayout<Float>.stride
    }

    /// `textureTransform` describes how `inputTexture` maps onto presentation
    /// pixels; pass identity if there's no transform. Encodes onto
    /// `commandBuffer` without committing it.
    public func encode(
        inputTexture: MTLTexture,
        textureTransform: simd_float4x4,
        outputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws
    {
        guard outputBuffer.length >= self.outputBufferLength else
        {
            throw EfficientTAMError("The EfficientTAM frame buffer is too small.")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            throw EfficientTAMError("Could not create the EfficientTAM frame preprocess encoder.")
        }
        var uniforms = Uniforms(
            outputSize: simd_uint2(UInt32(EfficientTAMImageEncoder.inputWidth), UInt32(EfficientTAMImageEncoder.inputHeight)),
            textureTransform: textureTransform
        )
        encoder.label = "EfficientTAM Frame Preprocess"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setTexture(inputTexture, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        EfficientTAMComputeLibrary.dispatch(
            encoder,
            pipeline: self.pipeline,
            width: EfficientTAMImageEncoder.inputWidth,
            height: EfficientTAMImageEncoder.inputHeight
        )
        encoder.endEncoding()
    }
}
