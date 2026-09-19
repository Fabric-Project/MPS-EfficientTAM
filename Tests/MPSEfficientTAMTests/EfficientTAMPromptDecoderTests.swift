import Foundation
import Metal
import Testing
@testable import MPSEfficientTAM

@Test func decoderMatchesOfficialPyTorchPointPrompt() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let decoder = try EfficientTAMPromptDecoder(
        promptCount: 2,
        commandQueue: commandQueue,
        maxFramesInFlight: 1
    )
    let embedding = try fixture(named: "encoder_reference")
    guard let embeddingBuffer = device.makeBuffer(
        bytes: embedding,
        length: embedding.count * MemoryLayout<Float>.stride
    ) else
    {
        return
    }
    let prompts = [
        EfficientTAMPrompt(x: 160, y: 240, label: .positivePoint),
        EfficientTAMPrompt(x: 0, y: 0, label: .padding),
    ]

    let prediction = try decoder.run(imageEmbeddingBuffer: embeddingBuffer, prompts: prompts)
    let referenceMasks = try fixture(named: "decoder_masks_reference")
    let referenceIoU = try fixture(named: "decoder_iou_reference")
    let referenceObjectScore = try fixture(named: "decoder_object_score_reference")
    let referenceObjectPointers = try fixture(named: "decoder_object_pointers_reference")
    #expect(prediction.maskLogits.count == referenceMasks.count)
    #expect(prediction.iouPredictions.count == referenceIoU.count)

    let maskErrors = zip(prediction.maskLogits, referenceMasks).map { abs($0 - $1) }
    let maskMeanAbsoluteError = maskErrors.reduce(0, +) / Float(maskErrors.count)
    let maskMaximumAbsoluteError = maskErrors.max() ?? .infinity
    let iouErrors = zip(prediction.iouPredictions, referenceIoU).map { abs($0 - $1) }
    let iouMaximumAbsoluteError = iouErrors.max() ?? .infinity
    let objectScoreError = abs(prediction.objectScoreLogit - referenceObjectScore[0])
    let objectPointerMaximumAbsoluteError = zip(prediction.objectPointers, referenceObjectPointers)
        .map { abs($0 - $1) }
        .max() ?? .infinity
    print(
        "Decoder mask MAE=\(maskMeanAbsoluteError), mask max=\(maskMaximumAbsoluteError), "
            + "IoU max=\(iouMaximumAbsoluteError)"
    )
    #expect(maskMeanAbsoluteError < 0.002)
    #expect(maskMaximumAbsoluteError < 0.03)
    #expect(iouMaximumAbsoluteError < 0.001)
    #expect(objectScoreError < 0.001)
    #expect(objectPointerMaximumAbsoluteError < 0.001)
}

@Test func rejectsWrongPromptCount() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let decoder = try EfficientTAMPromptDecoder(promptCount: 2, commandQueue: commandQueue)
    #expect(throws: EfficientTAMError.self) {
        try decoder.makePromptBuffers([
            EfficientTAMPrompt(x: 10, y: 20, label: .positivePoint),
        ])
    }
}

@Test func maskSelectorKeepsMaskAndPointerAligned() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let selector = try EfficientTAMMaskSelector(commandQueue: commandQueue, maxFramesInFlight: 1)
    let masks = try fixture(named: "decoder_masks_reference")
    let iou = try fixture(named: "decoder_iou_reference")
    let pointers = try fixture(named: "decoder_object_pointers_reference")
    guard let masksBuffer = device.makeBuffer(bytes: masks, length: masks.count * MemoryLayout<Float>.stride),
          let iouBuffer = device.makeBuffer(bytes: iou, length: iou.count * MemoryLayout<Float>.stride),
          let pointersBuffer = device.makeBuffer(bytes: pointers, length: pointers.count * MemoryLayout<Float>.stride) else
    {
        return
    }
    let selected = try selector.run(
        maskLogitsBuffer: masksBuffer,
        iouPredictionsBuffer: iouBuffer,
        objectPointersBuffer: pointersBuffer
    )
    let bestIndex = iou.indices.max(by: { iou[$0] < iou[$1] }) ?? 0
    let maskStart = bestIndex * 128 * 128
    let pointerStart = bestIndex * 256
    #expect(selected.iouPrediction == iou[bestIndex])
    #expect(selected.maskLogits == Array(masks[maskStart..<(maskStart + 128 * 128)]))
    #expect(selected.objectPointer == Array(pointers[pointerStart..<(pointerStart + 256)]))
}

@Test func encoderAndDecoderChainWithoutAnInterveningCPUWait() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue(),
          let encoderCommandBuffer = commandQueue.makeCommandBuffer(),
          let decoderCommandBuffer = commandQueue.makeCommandBuffer(),
          let verificationCommandBuffer = commandQueue.makeCommandBuffer() else
    {
        return
    }
    let encoder = try EfficientTAMImageEncoder(commandQueue: commandQueue, maxFramesInFlight: 1)
    let decoder = try EfficientTAMPromptDecoder(promptCount: 2, commandQueue: commandQueue, maxFramesInFlight: 1)
    let inputCount = EfficientTAMImageEncoder.inputWidth * EfficientTAMImageEncoder.inputHeight * 3
    let rgb = (0..<inputCount).map { Float($0 % 251) / 250 }
    let promptBuffers = try decoder.makePromptBuffers([
        EfficientTAMPrompt(x: 160, y: 240, label: .positivePoint),
        EfficientTAMPrompt(x: 0, y: 0, label: .padding),
    ])
    guard let inputBuffer = device.makeBuffer(
        bytes: rgb,
        length: rgb.count * MemoryLayout<Float>.stride
    ), let embeddingBuffer = device.makeBuffer(
        length: encoder.outputBufferLength,
        options: .storageModePrivate
    ), let maskBuffer = device.makeBuffer(
        length: decoder.maskLogitsBufferLength,
        options: .storageModePrivate
    ), let iouBuffer = device.makeBuffer(
        length: decoder.iouPredictionsBufferLength,
        options: .storageModePrivate
    ), let maskStaging = device.makeBuffer(
        length: decoder.maskLogitsBufferLength,
        options: .storageModeShared
    ), let iouStaging = device.makeBuffer(
        length: decoder.iouPredictionsBufferLength,
        options: .storageModeShared
    ) else
    {
        return
    }

    let encoderAccepted = try encoder.encode(
        inputBuffer: inputBuffer,
        outputBuffer: embeddingBuffer,
        commandBuffer: encoderCommandBuffer,
        commit: true
    )
    let decoderAccepted = try decoder.encode(
        imageEmbeddingBuffer: embeddingBuffer,
        promptCoordinatesBuffer: promptBuffers.coordinates,
        promptLabelsBuffer: promptBuffers.labels,
        maskLogitsBuffer: maskBuffer,
        iouPredictionsBuffer: iouBuffer,
        commandBuffer: decoderCommandBuffer,
        commit: true
    )
    #expect(encoderAccepted)
    #expect(decoderAccepted)

    guard let blit = verificationCommandBuffer.makeBlitCommandEncoder() else { return }
    blit.copy(
        from: maskBuffer,
        sourceOffset: 0,
        to: maskStaging,
        destinationOffset: 0,
        size: decoder.maskLogitsBufferLength
    )
    blit.copy(
        from: iouBuffer,
        sourceOffset: 0,
        to: iouStaging,
        destinationOffset: 0,
        size: decoder.iouPredictionsBufferLength
    )
    blit.endEncoding()
    verificationCommandBuffer.commit()
    verificationCommandBuffer.waitUntilCompleted()

    let maskValues = UnsafeBufferPointer(
        start: maskStaging.contents().assumingMemoryBound(to: Float.self),
        count: decoder.maskLogitsBufferLength / MemoryLayout<Float>.stride
    )
    let iouValues = UnsafeBufferPointer(
        start: iouStaging.contents().assumingMemoryBound(to: Float.self),
        count: EfficientTAMPromptDecoder.maskCount
    )
    #expect(maskValues.allSatisfy { $0.isFinite })
    #expect(iouValues.allSatisfy { $0.isFinite })
    let referenceIoU = try fixture(named: "decoder_iou_reference")
    let maximumIoUError = zip(iouValues, referenceIoU).map { abs($0 - $1) }.max() ?? .infinity
    #expect(maximumIoUError < 0.001)
}

private func fixture(named name: String) throws -> [Float]
{
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: "bin", subdirectory: "Fixtures")
    )
    return try Data(contentsOf: url).withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float.self))
    }
}
