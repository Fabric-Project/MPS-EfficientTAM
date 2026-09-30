import Foundation
import Metal
import Testing
@testable import MPSEfficientTAM

@Test func memoryEncoderMatchesOfficialPyTorch() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let precision = efficientTAMTestPrecision()
    let encoder = try EfficientTAMMemoryEncoder(commandQueue: commandQueue, maxFramesInFlight: 1, precision: precision)
    let imageEmbedding = try memoryFixture(named: "encoder_reference")
    let selectedMask = try memoryFixture(named: "memory_selected_mask_reference")
    guard let imageEmbeddingBuffer = device.makeBuffer(
        bytes: imageEmbedding,
        length: imageEmbedding.count * MemoryLayout<Float>.stride
    ), let selectedMaskBuffer = device.makeBuffer(
        bytes: selectedMask,
        length: selectedMask.count * MemoryLayout<Float>.stride
    ) else
    {
        return
    }

    let actual = try encoder.run(
        imageEmbeddingBuffer: imageEmbeddingBuffer,
        maskLogitsBuffer: selectedMaskBuffer
    )
    let reference = try memoryFixture(named: "memory_features_reference")
    let errors = zip(actual, reference).map { abs($0 - $1) }
    let meanAbsoluteError = errors.reduce(0, +) / Float(errors.count)
    let maximumAbsoluteError = errors.max() ?? .infinity
    print("Memory encoder \(precision) vs PyTorch: MAE=\(meanAbsoluteError), max=\(maximumAbsoluteError)")
    if precision == .float32
    {
        #expect(meanAbsoluteError < 0.001)
        #expect(maximumAbsoluteError < 0.02)
    }
    else
    {
        // Reduced tiers: 40 dB against the float32 encoder on the same inputs.
        let float32Output = try EfficientTAMMemoryEncoder(commandQueue: commandQueue, maxFramesInFlight: 1).run(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            maskLogitsBuffer: selectedMaskBuffer
        )
        let nonFiniteCount = actual.filter { !$0.isFinite }.count
        let psnr = efficientTAMPSNR(actual, float32Output)
        print("Memory encoder \(precision) vs float32: \(psnr.formatted(.number.precision(.fractionLength(1)))) dB, \(nonFiniteCount) non-finite")
        #expect(nonFiniteCount == 0)
        #expect(psnr > 40)
    }
}

@Test func memoryPositionEmbeddingMatchesOfficialPyTorch() throws
{
    let actual = EfficientTAMMemoryEncoder.positionEmbedding()
    let reference = try memoryFixture(named: "memory_position_reference")
    let errors = zip(actual, reference).map { abs($0 - $1) }
    let maximumAbsoluteError = errors.max() ?? .infinity
    #expect(actual.count == reference.count)
    #expect(maximumAbsoluteError < 0.00001)
}

private func memoryFixture(named name: String) throws -> [Float]
{
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: "bin", subdirectory: "Fixtures")
    )
    return try Data(contentsOf: url).withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float.self))
    }
}
