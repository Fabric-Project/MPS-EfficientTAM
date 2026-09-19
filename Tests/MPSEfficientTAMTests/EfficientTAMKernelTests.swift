import Foundation
import Metal
import simd
import Testing
@testable import MPSEfficientTAM

private let side = EfficientTAMImageEncoder.inputWidth

/// A 512x512 RGBA float texture whose texels encode their own coordinates, so
/// the packed output can be checked exactly.
private func gradientTexture(device: MTLDevice) -> MTLTexture?
{
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float,
        width: side,
        height: side,
        mipmapped: false
    )
    descriptor.usage = [.shaderRead]
    descriptor.storageMode = .shared
    guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
    var texels = [Float](repeating: 0, count: side * side * 4)
    for y in 0..<side
    {
        for x in 0..<side
        {
            let index = (y * side + x) * 4
            texels[index] = Float(x) / Float(side - 1)
            texels[index + 1] = Float(y) / Float(side - 1)
            texels[index + 2] = 0.25
            texels[index + 3] = 1
        }
    }
    texture.replace(
        region: MTLRegionMake2D(0, 0, side, side),
        mipmapLevel: 0,
        withBytes: texels,
        bytesPerRow: side * 4 * MemoryLayout<Float>.stride
    )
    return texture
}

private func runFramePreprocessor(
    device: MTLDevice,
    queue: MTLCommandQueue,
    transform: simd_float4x4
) throws -> [Float]
{
    let preprocessor = try EfficientTAMFramePreprocessor(device: device)
    let texture = try #require(gradientTexture(device: device))
    let output = try #require(device.makeBuffer(length: preprocessor.outputBufferLength, options: .storageModeShared))
    let commandBuffer = try #require(queue.makeCommandBuffer())
    try preprocessor.encode(inputTexture: texture, textureTransform: transform, outputBuffer: output, commandBuffer: commandBuffer)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return Array(UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: Float.self),
        count: preprocessor.outputBufferLength / MemoryLayout<Float>.stride
    ))
}

/// With an identity transform and a same-size texture, every output pixel
/// samples exactly one texel center, and no color conversion is applied.
@Test func framePreprocessorPacksTexelsUnchangedAsNHWC() throws
{
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
    let packed = try runFramePreprocessor(device: device, queue: queue, transform: matrix_identity_float4x4)
    for (x, y) in [(0, 0), (511, 0), (0, 511), (200, 300), (511, 511)]
    {
        let index = (y * side + x) * 3
        #expect(abs(packed[index] - Float(x) / Float(side - 1)) < 1e-6, "R at \(x),\(y)")
        #expect(abs(packed[index + 1] - Float(y) / Float(side - 1)) < 1e-6, "G at \(x),\(y)")
        #expect(abs(packed[index + 2] - 0.25) < 1e-6, "B at \(x),\(y)")
    }
}

/// The texture transform maps canonical coordinates to stored ones: a vertical
/// flip must put the texture's last row first.
@Test func framePreprocessorAppliesTheTextureTransform() throws
{
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
    let verticalFlip = simd_float4x4(columns: (
        simd_float4(1, 0, 0, 0),
        simd_float4(0, -1, 0, 0),
        simd_float4(0, 0, 1, 0),
        simd_float4(0, 1, 0, 1)
    ))
    let packed = try runFramePreprocessor(device: device, queue: queue, transform: verticalFlip)
    for (x, y) in [(0, 0), (200, 100), (511, 511)]
    {
        let index = (y * side + x) * 3
        #expect(abs(packed[index] - Float(x) / Float(side - 1)) < 1e-6, "R unchanged at \(x),\(y)")
        #expect(abs(packed[index + 1] - Float(side - 1 - y) / Float(side - 1)) < 1e-6, "G flipped at \(x),\(y)")
    }
}

@Test func maskProjectorWritesSigmoidAndRawLogitsAtNativeSize() throws
{
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
    let size = EfficientTAMMaskProjector.maskSize
    let logits = (0..<(size * size)).map { Float($0 % size) / Float(size - 1) * 16 - 8 }
    let logitsBuffer = try #require(device.makeBuffer(bytes: logits, length: logits.count * MemoryLayout<Float>.stride))
    let projector = try EfficientTAMMaskProjector(device: device)

    for applySigmoid in [true, false]
    {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: size, height: size, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let commandBuffer = try #require(queue.makeCommandBuffer())
        try projector.encode(maskLogitsBuffer: logitsBuffer, outputTexture: texture, applySigmoid: applySigmoid, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        var values = [Float](repeating: 0, count: size * size)
        texture.getBytes(&values, bytesPerRow: size * MemoryLayout<Float>.stride, from: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0)
        let maximumError = zip(values, logits).map
        {
            let expected = applySigmoid ? 1 / (1 + exp(-$1)) : $1
            return abs($0 - expected)
        }.max() ?? .infinity
        #expect(maximumError < 1e-5, "sigmoid=\(applySigmoid) max error \(maximumError)")
    }
}
