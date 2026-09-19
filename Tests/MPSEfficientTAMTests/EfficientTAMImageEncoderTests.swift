import Foundation
import Metal
import Testing
@testable import MPSEfficientTAM

@Test func rejectsWrongInputSize() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let encoder = try EfficientTAMImageEncoder(commandQueue: commandQueue)
    guard let buffer = device.makeBuffer(length: 16) else { return }
    #expect(throws: EfficientTAMError.self) {
        try encoder.run(inputBuffer: buffer)
    }
}

@Test func matchesOfficialPyTorchImageEncoder() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let encoder = try EfficientTAMImageEncoder(commandQueue: commandQueue, maxFramesInFlight: 1)
    let count = EfficientTAMImageEncoder.inputWidth * EfficientTAMImageEncoder.inputHeight * 3
    let rgb = (0..<count).map { Float($0 % 251) / 250 }
    guard let inputBuffer = device.makeBuffer(
        bytes: rgb,
        length: rgb.count * MemoryLayout<Float>.stride
    ) else
    {
        return
    }

    let actual = try encoder.run(inputBuffer: inputBuffer)
    let referenceURL = try #require(
        Bundle.module.url(forResource: "encoder_reference", withExtension: "bin", subdirectory: "Fixtures")
    )
    let referenceData = try Data(contentsOf: referenceURL)
    let reference = referenceData.withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float.self))
    }

    #expect(actual.count == reference.count)
    let absoluteErrors = zip(actual, reference).map { abs($0 - $1) }
    let meanAbsoluteError = absoluteErrors.reduce(0, +) / Float(absoluteErrors.count)
    let maximumAbsoluteError = absoluteErrors.max() ?? .infinity
    #expect(meanAbsoluteError < 0.0005)
    #expect(maximumAbsoluteError < 0.01)
}

@Test func encodeProducesReadableGPUOutputWithoutAnInterveningCPUWait() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue(),
          let modelCommandBuffer = commandQueue.makeCommandBuffer(),
          let verificationCommandBuffer = commandQueue.makeCommandBuffer(),
          let blit = verificationCommandBuffer.makeBlitCommandEncoder(),
          let inputBuffer = device.makeBuffer(
              length: EfficientTAMImageEncoder.inputWidth * EfficientTAMImageEncoder.inputHeight * 3 * MemoryLayout<Float>.stride
          ) else
    {
        return
    }
    let encoder = try EfficientTAMImageEncoder(commandQueue: commandQueue, maxFramesInFlight: 1)
    guard let outputBuffer = device.makeBuffer(length: encoder.outputBufferLength, options: .storageModePrivate),
          let stagingBuffer = device.makeBuffer(length: encoder.outputBufferLength, options: .storageModeShared) else
    {
        return
    }

    let accepted = try encoder.encode(
        inputBuffer: inputBuffer,
        outputBuffer: outputBuffer,
        commandBuffer: modelCommandBuffer,
        commit: true
    )
    #expect(accepted)
    blit.copy(from: outputBuffer, sourceOffset: 0, to: stagingBuffer, destinationOffset: 0, size: encoder.outputBufferLength)
    blit.endEncoding()
    verificationCommandBuffer.commit()
    verificationCommandBuffer.waitUntilCompleted()

    let values = UnsafeBufferPointer(
        start: stagingBuffer.contents().assumingMemoryBound(to: Float.self),
        count: encoder.outputBufferLength / MemoryLayout<Float>.stride
    )
    #expect(values.allSatisfy { $0.isFinite })
}
