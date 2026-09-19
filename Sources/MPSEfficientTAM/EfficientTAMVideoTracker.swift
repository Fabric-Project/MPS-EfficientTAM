import Foundation
import Metal

/// GPU-resident result of one submitted video-tracking frame. Buffers remain
/// valid for the lifetime of this value and can feed the mask postprocessor or
/// another Metal pipeline without readback.
public struct EfficientTAMVideoTrackingOutput
{
    public let frameIndex: Int
    public let maskLogitsBuffer: MTLBuffer
    public let iouPredictionBuffer: MTLBuffer
    public let objectScoreLogitBuffer: MTLBuffer
    public let objectPointerBuffer: MTLBuffer
}

/// A single-object forward EfficientTAM tracker built from the package's
/// independently usable GPU stages. Each submission commits ordered command
/// buffers without waiting for GPU or CPU readback.
public final class EfficientTAMVideoTracker
{
    public let initialPromptCount: Int
    public let maximumSpatialMemoryCount = 7
    public let maximumObjectPointerCount = 16

    private let commandQueue: MTLCommandQueue
    private let imageEncoder: EfficientTAMImageEncoder
    private let initialDecoder: EfficientTAMPromptDecoder
    private let trackingDecoder: EfficientTAMPromptDecoder
    private let maskSelector: EfficientTAMMaskSelector
    private let memoryEncoder: EfficientTAMMemoryEncoder
    private let trackingPromptCoordinatesBuffer: MTLBuffer
    private let trackingPromptLabelsBuffer: MTLBuffer
    private let submissionSemaphore: DispatchSemaphore
    private let stateLock = NSLock()
    private var memoryEntries: [MemoryEntry] = []
    private var nextFrameIndex = 0
    private var attentionCache: [AttentionShape: EfficientTAMMemoryAttention] = [:]

    private struct MemoryEntry
    {
        let frameIndex: Int
        let isConditioning: Bool
        let features: MTLBuffer
        let objectPointer: MTLBuffer
    }

    private struct AttentionShape: Hashable
    {
        let memoryCount: Int
        let pointerCount: Int
    }

    public init(
        initialPromptCount: Int = 2,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3
    ) throws
    {
        guard maxFramesInFlight > 0 else
        {
            throw EfficientTAMError("EfficientTAM maxFramesInFlight must be positive.")
        }
        self.initialPromptCount = initialPromptCount
        self.commandQueue = commandQueue
        self.submissionSemaphore = DispatchSemaphore(value: maxFramesInFlight)
        self.imageEncoder = try EfficientTAMImageEncoder(
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
        self.initialDecoder = try EfficientTAMPromptDecoder(
            promptCount: initialPromptCount,
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
        self.trackingDecoder = try EfficientTAMPromptDecoder(
            promptCount: 2,
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
        self.maskSelector = try EfficientTAMMaskSelector(
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
        self.memoryEncoder = try EfficientTAMMemoryEncoder(
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
        let trackingPrompts = try self.trackingDecoder.makePromptBuffers([
            EfficientTAMPrompt(x: 0, y: 0, label: .padding),
            EfficientTAMPrompt(x: 0, y: 0, label: .padding),
        ])
        self.trackingPromptCoordinatesBuffer = trackingPrompts.coordinates
        self.trackingPromptLabelsBuffer = trackingPrompts.labels
    }

    /// Removes all conditioning and tracked-frame memories. Already submitted
    /// GPU work remains valid, but subsequent frames start a new sequence.
    public func reset()
    {
        self.stateLock.lock()
        self.memoryEntries.removeAll(keepingCapacity: true)
        self.nextFrameIndex = 0
        self.stateLock.unlock()
    }

    /// Starts tracking from a prompted conditioning frame. Returns nil under
    /// normal in-flight backpressure instead of blocking.
    public func encodeInitialFrame(
        inputBuffer: MTLBuffer,
        prompts: [EfficientTAMPrompt]
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        let promptBuffers = try self.initialDecoder.makePromptBuffers(prompts)
        return try self.encodeInitialFrame(
            inputBuffer: inputBuffer,
            promptCoordinatesBuffer: promptBuffers.coordinates,
            promptLabelsBuffer: promptBuffers.labels
        )
    }

    public func encodeInitialFrame(
        inputBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        guard self.submissionSemaphore.wait(timeout: .now()) == .success else { return nil }
        var completionInstalled = false
        defer
        {
            if !completionInstalled { self.submissionSemaphore.signal() }
        }

        let frameIndex: Int
        self.stateLock.lock()
        guard self.memoryEntries.isEmpty else
        {
            self.stateLock.unlock()
            throw EfficientTAMError("Reset the EfficientTAM video tracker before starting another sequence.")
        }
        frameIndex = self.nextFrameIndex
        self.nextFrameIndex += 1
        self.stateLock.unlock()

        let buffers = try self.makeFrameBuffers()
        guard let encoderCommandBuffer = self.commandQueue.makeCommandBuffer(),
              let decoderCommandBuffer = self.commandQueue.makeCommandBuffer(),
              let selectorCommandBuffer = self.commandQueue.makeCommandBuffer(),
              let memoryCommandBuffer = self.commandQueue.makeCommandBuffer() else
        {
            throw EfficientTAMError("Could not create EfficientTAM video command buffers.")
        }
        guard try self.imageEncoder.encode(
            inputBuffer: inputBuffer,
            outputBuffer: buffers.imageEmbedding,
            commandBuffer: encoderCommandBuffer,
            commit: true
        ), try self.initialDecoder.encode(
            imageEmbeddingBuffer: buffers.imageEmbedding,
            promptCoordinatesBuffer: promptCoordinatesBuffer,
            promptLabelsBuffer: promptLabelsBuffer,
            maskLogitsBuffer: buffers.candidateMasks,
            iouPredictionsBuffer: buffers.candidateIoU,
            objectScoreLogitBuffer: buffers.objectScore,
            objectPointersBuffer: buffers.candidatePointers,
            commandBuffer: decoderCommandBuffer,
            commit: true
        ), try self.maskSelector.encode(
            maskLogitsBuffer: buffers.candidateMasks,
            iouPredictionsBuffer: buffers.candidateIoU,
            objectPointersBuffer: buffers.candidatePointers,
            selectedMaskLogitsBuffer: buffers.selectedMask,
            selectedIoUPredictionBuffer: buffers.selectedIoU,
            selectedObjectPointerBuffer: buffers.selectedPointer,
            commandBuffer: selectorCommandBuffer,
            commit: true
        ), try self.memoryEncoder.encode(
            imageEmbeddingBuffer: buffers.imageEmbedding,
            maskLogitsBuffer: buffers.selectedMask,
            memoryFeaturesBuffer: buffers.memoryFeatures,
            commandBuffer: memoryCommandBuffer,
            commit: false
        ) else
        {
            throw EfficientTAMError("An EfficientTAM video stage rejected the initial frame due to backpressure.")
        }
        memoryCommandBuffer.addCompletedHandler { [weak self] _ in self?.submissionSemaphore.signal() }
        completionInstalled = true
        memoryCommandBuffer.commit()
        self.appendMemory(
            MemoryEntry(
                frameIndex: frameIndex,
                isConditioning: true,
                features: buffers.memoryFeatures,
                objectPointer: buffers.selectedPointer
            )
        )
        return EfficientTAMVideoTrackingOutput(
            frameIndex: frameIndex,
            maskLogitsBuffer: buffers.selectedMask,
            iouPredictionBuffer: buffers.selectedIoU,
            objectScoreLogitBuffer: buffers.objectScore,
            objectPointerBuffer: buffers.selectedPointer
        )
    }

    /// Tracks the object into the next frame using the retained memory bank.
    /// The returned buffers are GPU-resident and no completion wait occurs.
    public func encodeNextFrame(inputBuffer: MTLBuffer) throws -> EfficientTAMVideoTrackingOutput?
    {
        guard self.submissionSemaphore.wait(timeout: .now()) == .success else { return nil }
        var completionInstalled = false
        defer
        {
            if !completionInstalled { self.submissionSemaphore.signal() }
        }

        let frameIndex: Int
        let entries: [MemoryEntry]
        self.stateLock.lock()
        guard !self.memoryEntries.isEmpty else
        {
            self.stateLock.unlock()
            throw EfficientTAMError("Encode a conditioning frame before tracking subsequent frames.")
        }
        frameIndex = self.nextFrameIndex
        self.nextFrameIndex += 1
        entries = self.memoryEntries
        self.stateLock.unlock()

        let snapshot = try self.makeSnapshot(from: entries)
        let attention = try self.memoryAttention(
            memoryCount: snapshot.spatialEntries.count,
            pointerCount: snapshot.pointerEntries.count
        )
        let positionBuffer = try attention.makeMemoryPositionBuffer(
            temporalPositionIndexes: snapshot.temporalPositionIndexes
        )
        let buffers = try self.makeFrameBuffers()
        guard let encoderCommandBuffer = self.commandQueue.makeCommandBuffer(),
              let attentionCommandBuffer = self.commandQueue.makeCommandBuffer(),
              let decoderCommandBuffer = self.commandQueue.makeCommandBuffer(),
              let selectorCommandBuffer = self.commandQueue.makeCommandBuffer(),
              let memoryCommandBuffer = self.commandQueue.makeCommandBuffer() else
        {
            throw EfficientTAMError("Could not create EfficientTAM video command buffers.")
        }
        guard try self.imageEncoder.encode(
            inputBuffer: inputBuffer,
            outputBuffer: buffers.imageEmbedding,
            commandBuffer: encoderCommandBuffer,
            commit: true
        ), try attention.encode(
            imageEmbeddingBuffer: buffers.imageEmbedding,
            memoryFeaturesBuffer: snapshot.memoryFeatures,
            memoryPositionBuffer: positionBuffer,
            objectPointersBuffer: snapshot.objectPointers,
            outputBuffer: buffers.conditionedEmbedding,
            commandBuffer: attentionCommandBuffer,
            commit: true
        ), try self.trackingDecoder.encode(
            imageEmbeddingBuffer: buffers.conditionedEmbedding,
            promptCoordinatesBuffer: self.trackingPromptCoordinatesBuffer,
            promptLabelsBuffer: self.trackingPromptLabelsBuffer,
            maskLogitsBuffer: buffers.candidateMasks,
            iouPredictionsBuffer: buffers.candidateIoU,
            objectScoreLogitBuffer: buffers.objectScore,
            objectPointersBuffer: buffers.candidatePointers,
            commandBuffer: decoderCommandBuffer,
            commit: true
        ), try self.maskSelector.encode(
            maskLogitsBuffer: buffers.candidateMasks,
            iouPredictionsBuffer: buffers.candidateIoU,
            objectPointersBuffer: buffers.candidatePointers,
            selectedMaskLogitsBuffer: buffers.selectedMask,
            selectedIoUPredictionBuffer: buffers.selectedIoU,
            selectedObjectPointerBuffer: buffers.selectedPointer,
            commandBuffer: selectorCommandBuffer,
            commit: true
        ), try self.memoryEncoder.encode(
            imageEmbeddingBuffer: buffers.imageEmbedding,
            maskLogitsBuffer: buffers.selectedMask,
            memoryFeaturesBuffer: buffers.memoryFeatures,
            commandBuffer: memoryCommandBuffer,
            commit: false
        ) else
        {
            throw EfficientTAMError("An EfficientTAM video stage rejected the frame due to backpressure.")
        }
        memoryCommandBuffer.addCompletedHandler { [weak self] _ in self?.submissionSemaphore.signal() }
        completionInstalled = true
        memoryCommandBuffer.commit()
        self.appendMemory(
            MemoryEntry(
                frameIndex: frameIndex,
                isConditioning: false,
                features: buffers.memoryFeatures,
                objectPointer: buffers.selectedPointer
            )
        )
        return EfficientTAMVideoTrackingOutput(
            frameIndex: frameIndex,
            maskLogitsBuffer: buffers.selectedMask,
            iouPredictionBuffer: buffers.selectedIoU,
            objectScoreLogitBuffer: buffers.objectScore,
            objectPointerBuffer: buffers.selectedPointer
        )
    }

    private struct FrameBuffers
    {
        let imageEmbedding: MTLBuffer
        let conditionedEmbedding: MTLBuffer
        let candidateMasks: MTLBuffer
        let candidateIoU: MTLBuffer
        let objectScore: MTLBuffer
        let candidatePointers: MTLBuffer
        let selectedMask: MTLBuffer
        let selectedIoU: MTLBuffer
        let selectedPointer: MTLBuffer
        let memoryFeatures: MTLBuffer
    }

    private func makeFrameBuffers() throws -> FrameBuffers
    {
        let device = self.commandQueue.device
        guard let imageEmbedding = device.makeBuffer(length: self.imageEncoder.outputBufferLength, options: .storageModePrivate),
              let conditionedEmbedding = device.makeBuffer(length: self.imageEncoder.outputBufferLength, options: .storageModePrivate),
              let candidateMasks = device.makeBuffer(length: self.initialDecoder.maskLogitsBufferLength, options: .storageModePrivate),
              let candidateIoU = device.makeBuffer(length: self.initialDecoder.iouPredictionsBufferLength, options: .storageModePrivate),
              let objectScore = device.makeBuffer(length: self.initialDecoder.objectScoreLogitBufferLength, options: .storageModePrivate),
              let candidatePointers = device.makeBuffer(length: self.initialDecoder.objectPointersBufferLength, options: .storageModePrivate),
              let selectedMask = device.makeBuffer(length: self.maskSelector.selectedMaskLogitsBufferLength, options: .storageModePrivate),
              let selectedIoU = device.makeBuffer(length: self.maskSelector.selectedIoUPredictionBufferLength, options: .storageModePrivate),
              let selectedPointer = device.makeBuffer(length: self.maskSelector.selectedObjectPointerBufferLength, options: .storageModePrivate),
              let memoryFeatures = device.makeBuffer(length: self.memoryEncoder.memoryFeaturesBufferLength, options: .storageModePrivate) else
        {
            throw EfficientTAMError("Could not allocate EfficientTAM video frame buffers.")
        }
        return FrameBuffers(
            imageEmbedding: imageEmbedding,
            conditionedEmbedding: conditionedEmbedding,
            candidateMasks: candidateMasks,
            candidateIoU: candidateIoU,
            objectScore: objectScore,
            candidatePointers: candidatePointers,
            selectedMask: selectedMask,
            selectedIoU: selectedIoU,
            selectedPointer: selectedPointer,
            memoryFeatures: memoryFeatures
        )
    }

    private struct MemorySnapshot
    {
        let spatialEntries: [MemoryEntry]
        let pointerEntries: [MemoryEntry]
        let temporalPositionIndexes: [Int]
        let memoryFeatures: MTLBuffer
        let objectPointers: MTLBuffer
    }

    private func makeSnapshot(from entries: [MemoryEntry]) throws -> MemorySnapshot
    {
        let conditioning = Array(entries.filter(\.isConditioning).suffix(1))
        let nonConditioning = entries.filter { !$0.isConditioning }
        let recentSpatial = Array(nonConditioning.suffix(self.maximumSpatialMemoryCount - conditioning.count))
        let spatialEntries = conditioning + recentSpatial
        let recentPointers = Array(nonConditioning.suffix(self.maximumObjectPointerCount - conditioning.count))
        let pointerEntries = conditioning + recentPointers
        let newestFrame = entries.map(\.frameIndex).max() ?? 0
        let temporalIndexes = spatialEntries.map
        {
            entry in
            if entry.isConditioning { return 6 }
            return min(5, max(0, newestFrame - entry.frameIndex))
        }
        let memoryLength = spatialEntries.count * self.memoryEncoder.memoryFeaturesBufferLength
        let pointerLength = pointerEntries.count * self.maskSelector.selectedObjectPointerBufferLength
        guard let memoryBuffer = self.commandQueue.device.makeBuffer(length: memoryLength, options: .storageModePrivate),
              let pointerBuffer = self.commandQueue.device.makeBuffer(length: pointerLength, options: .storageModePrivate),
              let commandBuffer = self.commandQueue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else
        {
            throw EfficientTAMError("Could not allocate or assemble EfficientTAM video memory.")
        }
        for (index, entry) in spatialEntries.enumerated()
        {
            blit.copy(
                from: entry.features,
                sourceOffset: 0,
                to: memoryBuffer,
                destinationOffset: index * self.memoryEncoder.memoryFeaturesBufferLength,
                size: self.memoryEncoder.memoryFeaturesBufferLength
            )
        }
        for (index, entry) in pointerEntries.enumerated()
        {
            blit.copy(
                from: entry.objectPointer,
                sourceOffset: 0,
                to: pointerBuffer,
                destinationOffset: index * self.maskSelector.selectedObjectPointerBufferLength,
                size: self.maskSelector.selectedObjectPointerBufferLength
            )
        }
        blit.endEncoding()
        commandBuffer.commit()
        return MemorySnapshot(
            spatialEntries: spatialEntries,
            pointerEntries: pointerEntries,
            temporalPositionIndexes: temporalIndexes,
            memoryFeatures: memoryBuffer,
            objectPointers: pointerBuffer
        )
    }

    private func memoryAttention(memoryCount: Int, pointerCount: Int) throws -> EfficientTAMMemoryAttention
    {
        let shape = AttentionShape(memoryCount: memoryCount, pointerCount: pointerCount)
        self.stateLock.lock()
        if let cached = self.attentionCache[shape]
        {
            self.stateLock.unlock()
            return cached
        }
        self.stateLock.unlock()
        let attention = try EfficientTAMMemoryAttention(
            memoryFrameCount: memoryCount,
            objectPointerCount: pointerCount,
            commandQueue: self.commandQueue,
            maxFramesInFlight: 3
        )
        self.stateLock.lock()
        self.attentionCache[shape] = attention
        self.stateLock.unlock()
        return attention
    }

    private func appendMemory(_ entry: MemoryEntry)
    {
        self.stateLock.lock()
        self.memoryEntries.append(entry)
        let conditioning = self.memoryEntries.filter(\.isConditioning)
        let recent = Array(self.memoryEntries.filter { !$0.isConditioning }.suffix(self.maximumObjectPointerCount))
        self.memoryEntries = conditioning + recent
        self.stateLock.unlock()
    }
}
