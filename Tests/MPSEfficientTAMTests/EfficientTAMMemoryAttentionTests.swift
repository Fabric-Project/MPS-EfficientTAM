import Foundation
import Metal
import Testing
@testable import MPSEfficientTAM

@Test func memoryAttentionMatchesOfficialPyTorch() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let attention = try EfficientTAMMemoryAttention(
        memoryFrameCount: 1,
        objectPointerCount: 1,
        commandQueue: commandQueue,
        maxFramesInFlight: 1
    )
    let image = try attentionFixture(named: "attention_current_embedding_reference")
    let memory = try attentionFixture(named: "attention_memory_features_reference")
    let position = try attentionFixture(named: "attention_memory_position_reference")
    let pointer = try attentionFixture(named: "attention_object_pointer_reference")
    let generatedPositionBuffer = try attention.makeMemoryPositionBuffer(temporalPositionIndexes: [0])
    let generatedPosition = Array(
        UnsafeBufferPointer(
            start: generatedPositionBuffer.contents().assumingMemoryBound(to: Float.self),
            count: position.count
        )
    )
    let positionMaximumError = zip(generatedPosition, position).map { abs($0 - $1) }.max() ?? .infinity
    #expect(positionMaximumError < 0.00001)
    guard let imageBuffer = device.makeBuffer(bytes: image, length: image.count * MemoryLayout<Float>.stride),
          let memoryBuffer = device.makeBuffer(bytes: memory, length: memory.count * MemoryLayout<Float>.stride),
          let positionBuffer = device.makeBuffer(bytes: position, length: position.count * MemoryLayout<Float>.stride),
          let pointerBuffer = device.makeBuffer(bytes: pointer, length: pointer.count * MemoryLayout<Float>.stride) else
    {
        return
    }

    let actual = try attention.run(
        imageEmbeddingBuffer: imageBuffer,
        memoryFeaturesBuffer: memoryBuffer,
        memoryPositionBuffer: positionBuffer,
        objectPointersBuffer: pointerBuffer
    )
    let reference = try attentionFixture(named: "attention_output_reference")
    let errors = zip(actual, reference).map { abs($0 - $1) }
    let meanAbsoluteError = errors.reduce(0, +) / Float(errors.count)
    let maximumAbsoluteError = errors.max() ?? .infinity
    print("Memory attention MAE=\(meanAbsoluteError), max=\(maximumAbsoluteError)")
    #expect(meanAbsoluteError < 0.002)
    #expect(maximumAbsoluteError < 0.03)
}

/// The masked graph has capacity for 7 memories and 16 pointers. Feeding the
/// one-memory/one-pointer fixture through it with junk in every unused slot
/// must reproduce the exact-shape PyTorch output: masked keys get zero weight.
@Test func maskedMemoryAttentionIgnoresUnusedSlots() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let attention = try EfficientTAMMemoryAttention(
        memoryFrameCount: 7,
        objectPointerCount: 16,
        usesKeyMask: true,
        commandQueue: commandQueue,
        maxFramesInFlight: 1
    )
    let image = try attentionFixture(named: "attention_current_embedding_reference")
    let memory = try attentionFixture(named: "attention_memory_features_reference")
    let position = try attentionFixture(named: "attention_memory_position_reference")
    let pointer = try attentionFixture(named: "attention_object_pointer_reference")

    let junk: Float = 3
    let paddedMemory = memory + [Float](repeating: junk, count: 6 * memory.count)
    let paddedPointers = pointer + [Float](repeating: junk, count: 15 * pointer.count)
    let generatedPositionBuffer = try attention.makeMemoryPositionBuffer(temporalPositionIndexes: [0])
    let generatedPosition = Array(
        UnsafeBufferPointer(
            start: generatedPositionBuffer.contents().assumingMemoryBound(to: Float.self),
            count: position.count
        )
    )
    #expect((zip(generatedPosition, position).map { abs($0 - $1) }.max() ?? .infinity) < 0.00001)

    guard let imageBuffer = device.makeBuffer(bytes: image, length: image.count * MemoryLayout<Float>.stride),
          let memoryBuffer = device.makeBuffer(bytes: paddedMemory, length: paddedMemory.count * MemoryLayout<Float>.stride),
          let pointerBuffer = device.makeBuffer(bytes: paddedPointers, length: paddedPointers.count * MemoryLayout<Float>.stride) else
    {
        return
    }
    let keyMask = try attention.makeKeyMaskBuffer(validMemoryFrameCount: 1, validPointerCount: 1)
    let actual = try attention.run(
        imageEmbeddingBuffer: imageBuffer,
        memoryFeaturesBuffer: memoryBuffer,
        memoryPositionBuffer: generatedPositionBuffer,
        objectPointersBuffer: pointerBuffer,
        keyMaskBuffer: keyMask
    )
    let reference = try attentionFixture(named: "attention_output_reference")
    let errors = zip(actual, reference).map { abs($0 - $1) }
    let meanAbsoluteError = errors.reduce(0, +) / Float(errors.count)
    let maximumAbsoluteError = errors.max() ?? .infinity
    print("Masked memory attention MAE=\(meanAbsoluteError), max=\(maximumAbsoluteError)")
    #expect(meanAbsoluteError < 0.002)
    #expect(maximumAbsoluteError < 0.03)
}

private func attentionFixture(named name: String) throws -> [Float]
{
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: "bin", subdirectory: "Fixtures")
    )
    return try Data(contentsOf: url).withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float.self))
    }
}
