import Foundation
import Metal
import MetalPerformanceShaders

/// GPU-resident result of one tracked frame. Buffers remain valid for the
/// lifetime of this value. They are written by the command buffer the frame was
/// encoded onto, so work encoded after it on that same buffer (or committed
/// after it on the same queue) can read them without any wait or readback.
public struct EfficientTAMVideoTrackingOutput
{
    public let frameIndex: Int
    public let maskLogitsBuffer: MTLBuffer
    public let iouPredictionBuffer: MTLBuffer
    public let objectScoreLogitBuffer: MTLBuffer
    public let objectPointerBuffer: MTLBuffer
    /// The frame's newly encoded spatial memory, `[1, 64, 32, 32]` float32 NCHW.
    public let memoryFeaturesBuffer: MTLBuffer
}

/// A single-object forward EfficientTAM tracker built from the package's
/// independently usable GPU stages.
///
/// Every stage of a frame (image encode, memory attention, decode, mask
/// selection, memory encode, and the memory-bank assembly blit) is encoded back
/// to back onto one command buffer. Pass Fabric's per-frame `MPSCommandBuffer`
/// with `commit: false` to make the whole frame part of it; the tracker never
/// commits or waits on a buffer it does not own. The overloads without a
/// command buffer use the tracker's own queue and commit.
///
/// The tracker's memory bank references buffers written by frames already
/// encoded. If a caller encodes a frame and then abandons the uncommitted
/// command buffer, call `reset()`: those buffers were never written.
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
    private let stageSlotCount: Int
    private let submissionSlots: EfficientTAMSlotPool
    private let stateLock = NSLock()
    private var memoryEntries: [MemoryEntry] = []
    private var nextFrameIndex = 0
    private var maskedAttention: EfficientTAMMemoryAttention?
    private var keyMaskCache: [AttentionShape: MTLBuffer] = [:]

    /// Test/benchmark hook: CPU milliseconds spent inside each part of an encode.
    var cpuTimingHandler: ((_ frameIndex: Int, _ stage: String, _ milliseconds: Double) -> Void)?

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

    private enum FrameKind
    {
        case initial(promptCoordinates: MTLBuffer, promptLabels: MTLBuffer)
        case tracking
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
        // The tracker's slot pool is the only admission control; its stages are
        // private, so their own slot pools must never reject a frame the
        // tracker has admitted. A stage releases its slot from a completion
        // handler that can lag the tracker's own release by a callback hop, so
        // the stages get headroom rather than an exact match.
        let stageSlotCount = maxFramesInFlight * 2
        self.stageSlotCount = stageSlotCount
        self.submissionSlots = EfficientTAMSlotPool(count: maxFramesInFlight)
        self.imageEncoder = try EfficientTAMImageEncoder(
            commandQueue: commandQueue,
            maxFramesInFlight: stageSlotCount
        )
        self.initialDecoder = try EfficientTAMPromptDecoder(
            promptCount: initialPromptCount,
            commandQueue: commandQueue,
            maxFramesInFlight: stageSlotCount
        )
        self.trackingDecoder = try EfficientTAMPromptDecoder(
            promptCount: 2,
            commandQueue: commandQueue,
            maxFramesInFlight: stageSlotCount
        )
        self.maskSelector = try EfficientTAMMaskSelector(
            commandQueue: commandQueue,
            maxFramesInFlight: stageSlotCount
        )
        self.memoryEncoder = try EfficientTAMMemoryEncoder(
            commandQueue: commandQueue,
            maxFramesInFlight: stageSlotCount
        )
        let trackingPrompts = try self.trackingDecoder.makePromptBuffers([
            EfficientTAMPrompt(x: 0, y: 0, label: .padding),
            EfficientTAMPrompt(x: 0, y: 0, label: .padding),
        ])
        self.trackingPromptCoordinatesBuffer = trackingPrompts.coordinates
        self.trackingPromptLabelsBuffer = trackingPrompts.labels
    }

    /// Removes all conditioning and tracked-frame memories. Already encoded GPU
    /// work remains valid, but subsequent frames start a new sequence.
    public func reset()
    {
        self.stateLock.lock()
        self.memoryEntries.removeAll(keepingCapacity: true)
        self.nextFrameIndex = 0
        self.stateLock.unlock()
    }

    // MARK: - Encode onto a caller's command buffer

    /// Starts tracking from a prompted conditioning frame, encoded onto
    /// `commandBuffer`. Returns nil under in-flight backpressure instead of
    /// blocking. With `commit: false` the caller commits the buffer; the
    /// tracker's in-flight slot is returned when that buffer completes.
    public func encodeInitialFrame(
        inputBuffer: MTLBuffer,
        prompts: [EfficientTAMPrompt],
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        let promptBuffers = try self.initialDecoder.makePromptBuffers(prompts)
        return try self.encodeInitialFrame(
            inputBuffer: inputBuffer,
            promptCoordinatesBuffer: promptBuffers.coordinates,
            promptLabelsBuffer: promptBuffers.labels,
            commandBuffer: commandBuffer,
            commit: commit
        )
    }

    public func encodeInitialFrame(
        inputBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        try self.encodeFrame(
            kind: .initial(promptCoordinates: promptCoordinatesBuffer, promptLabels: promptLabelsBuffer),
            inputBuffer: inputBuffer,
            commandBuffer: commandBuffer,
            commit: commit
        )
    }

    /// Tracks the object into the next frame using the retained memory bank,
    /// encoded onto `commandBuffer`. The returned buffers are GPU-resident and
    /// no wait occurs.
    public func encodeNextFrame(
        inputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        try self.encodeFrame(kind: .tracking, inputBuffer: inputBuffer, commandBuffer: commandBuffer, commit: commit)
    }

    // MARK: - Encode onto the tracker's own command buffer

    public func encodeInitialFrame(
        inputBuffer: MTLBuffer,
        prompts: [EfficientTAMPrompt]
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        let commandBuffer = try self.makeOwnedCommandBuffer()
        return try self.encodeInitialFrame(
            inputBuffer: inputBuffer,
            prompts: prompts,
            commandBuffer: commandBuffer,
            commit: true
        )
    }

    public func encodeInitialFrame(
        inputBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        let commandBuffer = try self.makeOwnedCommandBuffer()
        return try self.encodeInitialFrame(
            inputBuffer: inputBuffer,
            promptCoordinatesBuffer: promptCoordinatesBuffer,
            promptLabelsBuffer: promptLabelsBuffer,
            commandBuffer: commandBuffer,
            commit: true
        )
    }

    public func encodeNextFrame(inputBuffer: MTLBuffer) throws -> EfficientTAMVideoTrackingOutput?
    {
        let commandBuffer = try self.makeOwnedCommandBuffer()
        return try self.encodeNextFrame(inputBuffer: inputBuffer, commandBuffer: commandBuffer, commit: true)
    }

    private func makeOwnedCommandBuffer() throws -> MTLCommandBuffer
    {
        guard let rawCommandBuffer = self.commandQueue.makeCommandBuffer() else
        {
            throw EfficientTAMError("Could not create an EfficientTAM video command buffer.")
        }
        return MPSCommandBuffer(commandBuffer: rawCommandBuffer)
    }

    // MARK: - One frame

    private func encodeFrame(
        kind: FrameKind,
        inputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> EfficientTAMVideoTrackingOutput?
    {
        guard let submissionSlot = self.submissionSlots.tryAcquire() else { return nil }
        var completionInstalled = false
        var claimedFrameIndex: Int?
        defer
        {
            if !completionInstalled
            {
                self.releaseFrameClaim(claimedFrameIndex)
                self.submissionSlots.release(submissionSlot)
            }
        }

        let frameIndex: Int
        let entries: [MemoryEntry]
        self.stateLock.lock()
        switch kind
        {
        case .initial:
            guard self.memoryEntries.isEmpty else
            {
                self.stateLock.unlock()
                throw EfficientTAMError("Reset the EfficientTAM video tracker before starting another sequence.")
            }
        case .tracking:
            guard !self.memoryEntries.isEmpty else
            {
                self.stateLock.unlock()
                throw EfficientTAMError("Encode a conditioning frame before tracking subsequent frames.")
            }
        }
        frameIndex = self.nextFrameIndex
        self.nextFrameIndex += 1
        claimedFrameIndex = frameIndex
        entries = self.memoryEntries
        self.stateLock.unlock()

        let target = EfficientTAMCommandBuffer.target(for: commandBuffer)
        let encoding: MTLCommandBuffer = target.commandBuffer
        let buffers = try self.timedCPU(frameIndex, "frameBuffers") { try self.makeFrameBuffers() }

        var accepted = try self.timedCPU(frameIndex, "imageEncoder") {
            try self.imageEncoder.encode(
                inputBuffer: inputBuffer,
                outputBuffer: buffers.imageEmbedding,
                commandBuffer: encoding,
                commit: false
            )
        }
        switch kind
        {
        case .initial(let promptCoordinates, let promptLabels):
            accepted = try accepted && self.timedCPU(frameIndex, "decoder") {
                try self.initialDecoder.encode(
                    imageEmbeddingBuffer: buffers.imageEmbedding,
                    promptCoordinatesBuffer: promptCoordinates,
                    promptLabelsBuffer: promptLabels,
                    maskLogitsBuffer: buffers.candidateMasks,
                    iouPredictionsBuffer: buffers.candidateIoU,
                    objectScoreLogitBuffer: buffers.objectScore,
                    objectPointersBuffer: buffers.candidatePointers,
                    commandBuffer: encoding,
                    commit: false
                )
            }
        case .tracking:
            let attention = try self.memoryAttention()
            let snapshot = try self.timedCPU(frameIndex, "snapshot") {
                try self.encodeSnapshot(from: entries, attention: attention, onto: encoding)
            }
            let keyMask = try self.keyMask(
                attention: attention,
                memoryCount: snapshot.spatialEntryCount,
                pointerCount: snapshot.pointerEntryCount
            )
            accepted = try accepted && self.timedCPU(frameIndex, "memoryAttention") {
                try attention.encode(
                    imageEmbeddingBuffer: buffers.imageEmbedding,
                    memoryFeaturesBuffer: snapshot.memoryFeatures,
                    memoryPositionBuffer: snapshot.memoryPositions,
                    objectPointersBuffer: snapshot.objectPointers,
                    keyMaskBuffer: keyMask,
                    outputBuffer: buffers.conditionedEmbedding,
                    commandBuffer: encoding,
                    commit: false
                )
            }
            accepted = try accepted && self.timedCPU(frameIndex, "decoder") {
                try self.trackingDecoder.encode(
                    imageEmbeddingBuffer: buffers.conditionedEmbedding,
                    promptCoordinatesBuffer: self.trackingPromptCoordinatesBuffer,
                    promptLabelsBuffer: self.trackingPromptLabelsBuffer,
                    maskLogitsBuffer: buffers.candidateMasks,
                    iouPredictionsBuffer: buffers.candidateIoU,
                    objectScoreLogitBuffer: buffers.objectScore,
                    objectPointersBuffer: buffers.candidatePointers,
                    commandBuffer: encoding,
                    commit: false
                )
            }
        }
        accepted = try accepted && self.timedCPU(frameIndex, "selector") {
            try self.maskSelector.encode(
                maskLogitsBuffer: buffers.candidateMasks,
                iouPredictionsBuffer: buffers.candidateIoU,
                objectPointersBuffer: buffers.candidatePointers,
                selectedMaskLogitsBuffer: buffers.selectedMask,
                selectedIoUPredictionBuffer: buffers.selectedIoU,
                selectedObjectPointerBuffer: buffers.selectedPointer,
                commandBuffer: encoding,
                commit: false
            )
        }
        accepted = try accepted && self.timedCPU(frameIndex, "memoryEncoder") {
            try self.memoryEncoder.encode(
                imageEmbeddingBuffer: buffers.imageEmbedding,
                maskLogitsBuffer: buffers.selectedMask,
                memoryFeaturesBuffer: buffers.memoryFeatures,
                commandBuffer: encoding,
                commit: false
            )
        }
        guard accepted else
        {
            throw EfficientTAMError("An EfficientTAM video stage rejected the frame due to backpressure.")
        }

        // The tracker's own slot returns when the final underlying buffer
        // completes, which after an MPSGraph split is not the first one.
        try EfficientTAMCommandBuffer.finish(target, commit: commit) { [weak self] in
            self?.submissionSlots.release(submissionSlot)
        }
        completionInstalled = true

        let isConditioning: Bool
        if case .initial = kind { isConditioning = true } else { isConditioning = false }
        self.appendMemory(
            MemoryEntry(
                frameIndex: frameIndex,
                isConditioning: isConditioning,
                features: buffers.memoryFeatures,
                objectPointer: buffers.selectedPointer
            )
        )
        return EfficientTAMVideoTrackingOutput(
            frameIndex: frameIndex,
            maskLogitsBuffer: buffers.selectedMask,
            iouPredictionBuffer: buffers.selectedIoU,
            objectScoreLogitBuffer: buffers.objectScore,
            objectPointerBuffer: buffers.selectedPointer,
            memoryFeaturesBuffer: buffers.memoryFeatures
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
        let spatialEntryCount: Int
        let pointerEntryCount: Int
        let memoryFeatures: MTLBuffer
        let memoryPositions: MTLBuffer
        let objectPointers: MTLBuffer
    }

    /// Assembles the frame's memory bank with blits encoded onto the frame's own
    /// command buffer, so the copies are ordered after the frames that wrote the
    /// entries and before the attention that reads them.
    private func encodeSnapshot(
        from entries: [MemoryEntry],
        attention: EfficientTAMMemoryAttention,
        onto commandBuffer: MTLCommandBuffer
    ) throws -> MemorySnapshot
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
        // The masked attention graph always consumes full-capacity buffers.
        // Valid entries are packed first; the unused tail is zeroed because
        // masked keys still flow through the projections and must be finite.
        let memoryLength = attention.memoryFeaturesBufferLength
        let pointerLength = attention.objectPointersBufferLength
        guard let memoryBuffer = self.commandQueue.device.makeBuffer(length: memoryLength, options: .storageModePrivate),
              let positionBuffer = self.commandQueue.device.makeBuffer(length: memoryLength, options: .storageModePrivate),
              let pointerBuffer = self.commandQueue.device.makeBuffer(length: pointerLength, options: .storageModePrivate),
              let positionBlocks = attention.positionBlocks,
              let blit = commandBuffer.makeBlitCommandEncoder() else
        {
            throw EfficientTAMError("Could not allocate or assemble EfficientTAM video memory.")
        }
        let featureLength = self.memoryEncoder.memoryFeaturesBufferLength
        let pointerEntryLength = self.maskSelector.selectedObjectPointerBufferLength
        let memoryTail = spatialEntries.count * featureLength
        if memoryTail < memoryLength
        {
            blit.fill(buffer: memoryBuffer, range: memoryTail..<memoryLength, value: 0)
            blit.fill(buffer: positionBuffer, range: memoryTail..<memoryLength, value: 0)
        }
        let pointerTail = pointerEntries.count * pointerEntryLength
        if pointerTail < pointerLength
        {
            blit.fill(buffer: pointerBuffer, range: pointerTail..<pointerLength, value: 0)
        }
        for (index, temporalIndex) in temporalIndexes.enumerated()
        {
            blit.copy(
                from: positionBlocks,
                sourceOffset: temporalIndex * EfficientTAMMemoryAttention.positionBlockLength,
                to: positionBuffer,
                destinationOffset: index * featureLength,
                size: featureLength
            )
        }
        for (index, entry) in spatialEntries.enumerated()
        {
            blit.copy(from: entry.features, sourceOffset: 0, to: memoryBuffer, destinationOffset: index * featureLength, size: featureLength)
        }
        for (index, entry) in pointerEntries.enumerated()
        {
            blit.copy(from: entry.objectPointer, sourceOffset: 0, to: pointerBuffer, destinationOffset: index * pointerEntryLength, size: pointerEntryLength)
        }
        blit.endEncoding()
        return MemorySnapshot(
            spatialEntryCount: spatialEntries.count,
            pointerEntryCount: pointerEntries.count,
            memoryFeatures: memoryBuffer,
            memoryPositions: positionBuffer,
            objectPointers: pointerBuffer
        )
    }

    private func timedCPU<T>(_ frameIndex: Int, _ stage: String, _ body: () throws -> T) rethrows -> T
    {
        guard let handler = self.cpuTimingHandler else { return try body() }
        let start = CFAbsoluteTimeGetCurrent()
        defer { handler(frameIndex, stage, (CFAbsoluteTimeGetCurrent() - start) * 1000) }
        return try body()
    }

    /// A submission that fails before its memory is appended must not leave a
    /// hole in the frame numbering, which drives temporal position indexes.
    private func releaseFrameClaim(_ frameIndex: Int?)
    {
        guard let frameIndex else { return }
        self.stateLock.lock()
        if self.nextFrameIndex == frameIndex + 1 { self.nextFrameIndex = frameIndex }
        self.stateLock.unlock()
    }

    /// Compiles the memory-attention graph so that no submission pays a compile
    /// hitch on its first tracked frame. One graph with a key mask serves every
    /// memory and pointer count, so this is a single compile.
    public func prewarmMemoryAttention() throws
    {
        _ = try self.memoryAttention()
    }

    private func memoryAttention() throws -> EfficientTAMMemoryAttention
    {
        self.stateLock.lock()
        defer { self.stateLock.unlock() }
        if let maskedAttention = self.maskedAttention { return maskedAttention }
        let attention = try EfficientTAMMemoryAttention(
            memoryFrameCount: self.maximumSpatialMemoryCount,
            objectPointerCount: self.maximumObjectPointerCount,
            usesKeyMask: true,
            commandQueue: self.commandQueue,
            maxFramesInFlight: self.stageSlotCount
        )
        self.maskedAttention = attention
        return attention
    }

    private func keyMask(
        attention: EfficientTAMMemoryAttention,
        memoryCount: Int,
        pointerCount: Int
    ) throws -> MTLBuffer
    {
        let shape = AttentionShape(memoryCount: memoryCount, pointerCount: pointerCount)
        self.stateLock.lock()
        if let cached = self.keyMaskCache[shape]
        {
            self.stateLock.unlock()
            return cached
        }
        self.stateLock.unlock()
        let mask = try attention.makeKeyMaskBuffer(
            validMemoryFrameCount: memoryCount,
            validPointerCount: pointerCount
        )
        self.stateLock.lock()
        self.keyMaskCache[shape] = mask
        self.stateLock.unlock()
        return mask
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
