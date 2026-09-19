import Foundation
import Metal
import Testing
@testable import MPSEfficientTAM

@Test func maskPromptEncoderMatchesOfficialPyTorch() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let encoder = try EfficientTAMMaskPromptEncoder(commandQueue: commandQueue, maxFramesInFlight: 1)
    let priorMask = iterativePriorMask()
    guard let priorMaskBuffer = device.makeBuffer(
        bytes: priorMask,
        length: priorMask.count * MemoryLayout<Float>.stride
    ) else
    {
        return
    }

    let actual = try encoder.run(maskLogitsBuffer: priorMaskBuffer)
    let reference = try iterativeFixture(named: "mask_prompt_embedding_reference")
    let errors = zip(actual, reference).map { abs($0 - $1) }
    let meanAbsoluteError = errors.reduce(0, +) / Float(errors.count)
    let maximumAbsoluteError = errors.max() ?? .infinity
    print("Mask-prompt embedding MAE=\(meanAbsoluteError), max=\(maximumAbsoluteError)")
    #expect(meanAbsoluteError < 0.0002)
    #expect(maximumAbsoluteError < 0.003)
}

@Test func iterativePromptDecoderMatchesOfficialPyTorch() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let maskPromptEncoder = try EfficientTAMMaskPromptEncoder(commandQueue: commandQueue, maxFramesInFlight: 1)
    let decoder = try EfficientTAMPromptDecoder(promptCount: 2, commandQueue: commandQueue, maxFramesInFlight: 1)
    let priorMask = iterativePriorMask()
    let imageEmbedding = try iterativeFixture(named: "encoder_reference")
    guard let priorMaskBuffer = device.makeBuffer(
        bytes: priorMask,
        length: priorMask.count * MemoryLayout<Float>.stride
    ), let imageEmbeddingBuffer = device.makeBuffer(
        bytes: imageEmbedding,
        length: imageEmbedding.count * MemoryLayout<Float>.stride
    ), let densePromptBuffer = device.makeBuffer(
        length: maskPromptEncoder.outputBufferLength,
        options: .storageModeShared
    ), let commandBuffer = commandQueue.makeCommandBuffer() else
    {
        return
    }
    let accepted = try maskPromptEncoder.encode(
        maskLogitsBuffer: priorMaskBuffer,
        densePromptEmbeddingBuffer: densePromptBuffer,
        commandBuffer: commandBuffer,
        commit: true
    )
    #expect(accepted)
    commandBuffer.waitUntilCompleted()

    let prediction = try decoder.run(
        imageEmbeddingBuffer: imageEmbeddingBuffer,
        prompts: [
            EfficientTAMPrompt(x: 160, y: 240, label: .positivePoint),
            EfficientTAMPrompt(x: 0, y: 0, label: .padding),
        ],
        densePromptEmbeddingBuffer: densePromptBuffer
    )
    let referenceMasks = try iterativeFixture(named: "iterative_masks_reference")
    let referenceIoU = try iterativeFixture(named: "iterative_iou_reference")
    let maskErrors = zip(prediction.maskLogits, referenceMasks).map { abs($0 - $1) }
    let maskMeanAbsoluteError = maskErrors.reduce(0, +) / Float(maskErrors.count)
    let maskMaximumAbsoluteError = maskErrors.max() ?? .infinity
    let iouMaximumAbsoluteError = zip(prediction.iouPredictions, referenceIoU)
        .map { abs($0 - $1) }
        .max() ?? .infinity
    print(
        "Iterative mask MAE=\(maskMeanAbsoluteError), max=\(maskMaximumAbsoluteError), "
            + "IoU max=\(iouMaximumAbsoluteError)"
    )
    #expect(maskMeanAbsoluteError < 0.002)
    #expect(maskMaximumAbsoluteError < 0.03)
    #expect(iouMaximumAbsoluteError < 0.001)
}

@Test func maskPostprocessorMatchesOfficialPyTorchBilinearResize() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let postprocessor = try EfficientTAMMaskPostprocessor(
        outputWidth: 257,
        outputHeight: 193,
        commandQueue: commandQueue,
        maxFramesInFlight: 1
    )
    let input = try iterativeFixture(named: "iterative_masks_reference")
    guard let inputBuffer = device.makeBuffer(
        bytes: input,
        length: input.count * MemoryLayout<Float>.stride
    ) else
    {
        return
    }
    let actual = try postprocessor.run(maskLogitsBuffer: inputBuffer)
    let reference = try iterativeFixture(named: "resized_masks_reference")
    let errors = zip(actual, reference).map { abs($0 - $1) }
    let meanAbsoluteError = errors.reduce(0, +) / Float(errors.count)
    let maximumAbsoluteError = errors.max() ?? .infinity
    print("Postprocessed mask MAE=\(meanAbsoluteError), max=\(maximumAbsoluteError)")
    #expect(meanAbsoluteError < 0.0001)
    #expect(maximumAbsoluteError < 0.002)
}

@Test func iterativeStagesChainOnGPUWithoutInterveningCPUWaits() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue(),
          let maskPromptCommandBuffer = commandQueue.makeCommandBuffer(),
          let decoderCommandBuffer = commandQueue.makeCommandBuffer(),
          let postprocessCommandBuffer = commandQueue.makeCommandBuffer(),
          let verificationCommandBuffer = commandQueue.makeCommandBuffer() else
    {
        return
    }
    let maskPromptEncoder = try EfficientTAMMaskPromptEncoder(commandQueue: commandQueue, maxFramesInFlight: 1)
    let decoder = try EfficientTAMPromptDecoder(promptCount: 2, commandQueue: commandQueue, maxFramesInFlight: 1)
    let postprocessor = try EfficientTAMMaskPostprocessor(
        outputWidth: 257,
        outputHeight: 193,
        commandQueue: commandQueue,
        maxFramesInFlight: 1
    )
    let priorMask = iterativePriorMask()
    let imageEmbedding = try iterativeFixture(named: "encoder_reference")
    let promptBuffers = try decoder.makePromptBuffers([
        EfficientTAMPrompt(x: 160, y: 240, label: .positivePoint),
        EfficientTAMPrompt(x: 0, y: 0, label: .padding),
    ])
    guard let priorMaskBuffer = device.makeBuffer(
        bytes: priorMask,
        length: priorMask.count * MemoryLayout<Float>.stride
    ), let imageEmbeddingBuffer = device.makeBuffer(
        bytes: imageEmbedding,
        length: imageEmbedding.count * MemoryLayout<Float>.stride
    ), let densePromptBuffer = device.makeBuffer(
        length: maskPromptEncoder.outputBufferLength,
        options: .storageModePrivate
    ), let maskBuffer = device.makeBuffer(
        length: decoder.maskLogitsBufferLength,
        options: .storageModePrivate
    ), let iouBuffer = device.makeBuffer(
        length: decoder.iouPredictionsBufferLength,
        options: .storageModePrivate
    ), let resizedBuffer = device.makeBuffer(
        length: postprocessor.outputBufferLength,
        options: .storageModePrivate
    ), let stagingBuffer = device.makeBuffer(
        length: postprocessor.outputBufferLength,
        options: .storageModeShared
    ) else
    {
        return
    }

    #expect(try maskPromptEncoder.encode(
        maskLogitsBuffer: priorMaskBuffer,
        densePromptEmbeddingBuffer: densePromptBuffer,
        commandBuffer: maskPromptCommandBuffer,
        commit: true
    ))
    #expect(try decoder.encode(
        imageEmbeddingBuffer: imageEmbeddingBuffer,
        promptCoordinatesBuffer: promptBuffers.coordinates,
        promptLabelsBuffer: promptBuffers.labels,
        densePromptEmbeddingBuffer: densePromptBuffer,
        maskLogitsBuffer: maskBuffer,
        iouPredictionsBuffer: iouBuffer,
        commandBuffer: decoderCommandBuffer,
        commit: true
    ))
    #expect(try postprocessor.encode(
        maskLogitsBuffer: maskBuffer,
        resizedMaskLogitsBuffer: resizedBuffer,
        commandBuffer: postprocessCommandBuffer,
        commit: true
    ))

    guard let blit = verificationCommandBuffer.makeBlitCommandEncoder() else { return }
    blit.copy(
        from: resizedBuffer,
        sourceOffset: 0,
        to: stagingBuffer,
        destinationOffset: 0,
        size: postprocessor.outputBufferLength
    )
    blit.endEncoding()
    verificationCommandBuffer.commit()
    verificationCommandBuffer.waitUntilCompleted()

    let actual = UnsafeBufferPointer(
        start: stagingBuffer.contents().assumingMemoryBound(to: Float.self),
        count: postprocessor.outputBufferLength / MemoryLayout<Float>.stride
    )
    let reference = try iterativeFixture(named: "resized_masks_reference")
    let maximumAbsoluteError = zip(actual, reference).map { abs($0 - $1) }.max() ?? .infinity
    #expect(maximumAbsoluteError < 0.03)
}

private func iterativePriorMask() -> [Float]
{
    (0..<(128 * 128)).map { Float($0 % 113) / 29 - 1.5 }
}

private func iterativeFixture(named name: String) throws -> [Float]
{
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: "bin", subdirectory: "Fixtures")
    )
    return try Data(contentsOf: url).withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float.self))
    }
}
