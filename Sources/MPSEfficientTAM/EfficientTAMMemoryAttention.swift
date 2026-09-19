import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Conditions a frame embedding on a fixed number of spatial memories and
/// object pointers using EfficientTAM's four-layer RoPE memory transformer.
/// By default an instance compiles for exact memory and pointer counts. With
/// `usesKeyMask`, the counts are capacities: one compiled graph serves any
/// number of valid memories and pointers up to them, with unused slots removed
/// from attention by an additive key mask (see `makeKeyMaskBuffer`). Compiled
/// graphs embed their own copy of the weights, so a single masked instance is
/// far cheaper in memory than one exact instance per shape.
public final class EfficientTAMMemoryAttention
{
    public let memoryFrameCount: Int
    public let objectPointerCount: Int
    public let usesKeyMask: Bool

    public var imageEmbeddingBufferLength: Int { 256 * 32 * 32 * MemoryLayout<Float>.stride }
    public var memoryFeaturesBufferLength: Int
    {
        self.memoryFrameCount * 64 * 32 * 32 * MemoryLayout<Float>.stride
    }
    public var memoryPositionBufferLength: Int { self.memoryFeaturesBufferLength }
    public var objectPointersBufferLength: Int
    {
        self.objectPointerCount * 256 * MemoryLayout<Float>.stride
    }
    public var outputBufferLength: Int { self.imageEmbeddingBufferLength }

    private let graph = MPSGraph()
    private let commandQueue: MTLCommandQueue
    private let imageTensor: MPSGraphTensor
    private let memoryTensor: MPSGraphTensor
    private let memoryPositionTensor: MPSGraphTensor
    private let objectPointersTensor: MPSGraphTensor?
    private let keyMaskTensor: MPSGraphTensor?
    private let outputTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable
    private let slotPool: EfficientTAMSlotPool
    private let outputCaches: [OutputCache]
    private let temporalPositionEmbeddings: [Float]
    private let feedOrder: [MPSGraphTensor]
    /// The seven possible per-frame position blocks (spatial + temporal index
    /// 0...6), each `[64, 32, 32]`, so a masked instance can assemble a frame's
    /// position tensor with blits instead of CPU math. Nil unless `usesKeyMask`.
    let positionBlocks: MTLBuffer?
    static let positionBlockLength = 64 * 32 * 32 * MemoryLayout<Float>.stride

    private final class OutputCache
    {
        var values = [Float](repeating: 0, count: 256 * 32 * 32)
    }

    public convenience init(
        memoryFrameCount: Int,
        objectPointerCount: Int,
        usesKeyMask: Bool = false,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3
    ) throws
    {
        guard let binaryURL = Bundle.module.url(
            forResource: "EfficientTAMTiny512_weights",
            withExtension: "bin",
            subdirectory: "Models"
        ), let manifestURL = Bundle.module.url(
            forResource: "EfficientTAMTiny512_weights",
            withExtension: "json",
            subdirectory: "Models"
        ) else
        {
            throw EfficientTAMError("The bundled EfficientTAM Tiny 512 weights are missing.")
        }
        try self.init(
            weightsBinaryURL: binaryURL,
            weightsManifestURL: manifestURL,
            memoryFrameCount: memoryFrameCount,
            objectPointerCount: objectPointerCount,
            usesKeyMask: usesKeyMask,
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
    }

    public init(
        weightsBinaryURL: URL,
        weightsManifestURL: URL,
        memoryFrameCount: Int,
        objectPointerCount: Int,
        usesKeyMask: Bool = false,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3
    ) throws
    {
        guard memoryFrameCount > 0, objectPointerCount >= 0 else
        {
            throw EfficientTAMError("EfficientTAM memory attention requires memory and nonnegative pointer counts.")
        }
        guard maxFramesInFlight > 0 else
        {
            throw EfficientTAMError("EfficientTAM maxFramesInFlight must be positive.")
        }
        self.memoryFrameCount = memoryFrameCount
        self.objectPointerCount = objectPointerCount
        self.usesKeyMask = usesKeyMask
        self.commandQueue = commandQueue
        self.slotPool = EfficientTAMSlotPool(count: maxFramesInFlight)
        self.outputCaches = (0..<maxFramesInFlight).map { _ in OutputCache() }

        let weights = try EfficientTAMWeights(binaryURL: weightsBinaryURL, manifestURL: weightsManifestURL)
        let temporalEmbeddings = try weights.floats(named: "maskmem_tpos_enc")
        self.temporalPositionEmbeddings = temporalEmbeddings
        if usesKeyMask
        {
            let spatial = EfficientTAMMemoryEncoder.positionEmbedding()
            let pixelCount = SelfMemoryShape.pixelCount
            var blocks = [Float](repeating: 0, count: 7 * spatial.count)
            for index in 0..<7
            {
                for channel in 0..<EfficientTAMMemoryEncoder.memoryChannels
                {
                    let temporal = temporalEmbeddings[index * EfficientTAMMemoryEncoder.memoryChannels + channel]
                    for pixel in 0..<pixelCount
                    {
                        blocks[index * spatial.count + channel * pixelCount + pixel]
                            = spatial[channel * pixelCount + pixel] + temporal
                    }
                }
            }
            guard let buffer = commandQueue.device.makeBuffer(
                bytes: blocks,
                length: blocks.count * MemoryLayout<Float>.stride
            ) else
            {
                throw EfficientTAMError("Could not allocate the EfficientTAM position-block buffer.")
            }
            self.positionBlocks = buffer
        }
        else
        {
            self.positionBlocks = nil
        }
        let image = self.graph.placeholder(shape: [1, 256, 32, 32], dataType: .float32, name: "image_embedding")
        let memory = self.graph.placeholder(
            shape: [1, memoryFrameCount as NSNumber, 64, 32, 32],
            dataType: .float32,
            name: "memory_features"
        )
        let memoryPosition = self.graph.placeholder(
            shape: [1, memoryFrameCount as NSNumber, 64, 32, 32],
            dataType: .float32,
            name: "memory_position"
        )
        let objectPointers: MPSGraphTensor?
        if objectPointerCount > 0
        {
            objectPointers = self.graph.placeholder(
                shape: [1, objectPointerCount as NSNumber, 256],
                dataType: .float32,
                name: "object_pointers"
            )
        }
        else
        {
            objectPointers = nil
        }
        let keyMask: MPSGraphTensor?
        if usesKeyMask
        {
            keyMask = self.graph.placeholder(
                shape: [1, 1, (memoryFrameCount * 1024 + objectPointerCount * 4) as NSNumber],
                dataType: .float32,
                name: "key_mask"
            )
        }
        else
        {
            keyMask = nil
        }
        self.keyMaskTensor = keyMask
        self.imageTensor = image
        self.memoryTensor = memory
        self.memoryPositionTensor = memoryPosition
        self.objectPointersTensor = objectPointers
        let builder = EfficientTAMMemoryAttentionGraphBuilder(
            graph: self.graph,
            weights: weights,
            memoryFrameCount: memoryFrameCount,
            objectPointerCount: objectPointerCount
        )
        self.outputTensor = try builder.build(
            imageEmbedding: image,
            memoryFeatures: memory,
            memoryPosition: memoryPosition,
            objectPointers: objectPointers,
            keyMask: keyMask
        )

        let device = MPSGraphDevice(mtlDevice: commandQueue.device)
        let imageType = MPSGraphShapedType(shape: image.shape ?? [], dataType: .float32)
        let memoryType = MPSGraphShapedType(shape: memory.shape ?? [], dataType: .float32)
        let memoryPositionType = MPSGraphShapedType(shape: memoryPosition.shape ?? [], dataType: .float32)
        var feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            image: imageType,
            memory: memoryType,
            memoryPosition: memoryPositionType,
        ]
        if let objectPointers
        {
            feeds[objectPointers] = MPSGraphShapedType(shape: objectPointers.shape ?? [], dataType: .float32)
        }
        if let keyMask
        {
            feeds[keyMask] = MPSGraphShapedType(shape: keyMask.shape ?? [], dataType: .float32)
        }
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level1
        descriptor.waitForCompilationCompletion = true
        let compiled = self.graph.compile(
            with: device,
            feeds: feeds,
            targetTensors: [self.outputTensor],
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        // The executable defines its own feed order; follow it rather than
        // assuming dictionary or declaration order.
        let feedOrder = compiled.feedTensors ?? [image, memory, memoryPosition] + [objectPointers, keyMask].compactMap { $0 }
        self.feedOrder = feedOrder
        self.executable = compiled
        compiled.specialize(
            with: device,
            inputTypes: feedOrder.compactMap { feeds[$0] },
            compilationDescriptor: descriptor
        )
    }

    /// Builds the spatial-plus-temporal position buffer for the memories in
    /// this executable. Index 6 is a conditioning frame; indexes 0...5 are
    /// non-conditioning memories from newest to oldest.
    public func makeMemoryPositionBuffer(temporalPositionIndexes: [Int]) throws -> MTLBuffer
    {
        let validCount = self.usesKeyMask
            ? (1...self.memoryFrameCount).contains(temporalPositionIndexes.count)
            : temporalPositionIndexes.count == self.memoryFrameCount
        guard validCount,
              temporalPositionIndexes.allSatisfy({ (0..<7).contains($0) }) else
        {
            throw EfficientTAMError("Memory temporal-position indexes must match the compiled frame count and be 0...6.")
        }
        let spatial = EfficientTAMMemoryEncoder.positionEmbedding()
        let spatialPixelCount = SelfMemoryShape.pixelCount
        var values = [Float](repeating: 0, count: self.memoryFrameCount * spatial.count)
        for frame in 0..<temporalPositionIndexes.count
        {
            let temporalIndex = temporalPositionIndexes[frame]
            for channel in 0..<EfficientTAMMemoryEncoder.memoryChannels
            {
                let temporal = self.temporalPositionEmbeddings[
                    temporalIndex * EfficientTAMMemoryEncoder.memoryChannels + channel
                ]
                let sourceOffset = channel * spatialPixelCount
                let destinationOffset = frame * spatial.count + sourceOffset
                for pixel in 0..<spatialPixelCount
                {
                    values[destinationOffset + pixel] = spatial[sourceOffset + pixel] + temporal
                }
            }
        }
        guard let buffer = self.commandQueue.device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride
        ) else
        {
            throw EfficientTAMError("Could not allocate the EfficientTAM memory-position buffer.")
        }
        return buffer
    }

    /// Builds the additive key mask for a `usesKeyMask` instance: zero for the
    /// first `validMemoryFrameCount` memories and first `validPointerCount`
    /// pointers, a large negative value for every unused slot. Valid memories
    /// and pointers must be packed first in their buffers, and the unused tail
    /// of each buffer must hold finite values (zeros).
    public func makeKeyMaskBuffer(validMemoryFrameCount: Int, validPointerCount: Int) throws -> MTLBuffer
    {
        guard self.usesKeyMask,
              (1...self.memoryFrameCount).contains(validMemoryFrameCount),
              (0...self.objectPointerCount).contains(validPointerCount) else
        {
            throw EfficientTAMError("The key mask requires a key-mask instance and valid counts within its capacity.")
        }
        let spatialKeyCount = self.memoryFrameCount * 1024
        var values = [Float](repeating: -1e9, count: spatialKeyCount + self.objectPointerCount * 4)
        for index in 0..<(validMemoryFrameCount * 1024) { values[index] = 0 }
        for index in 0..<(validPointerCount * 4) { values[spatialKeyCount + index] = 0 }
        guard let buffer = self.commandQueue.device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride
        ) else
        {
            throw EfficientTAMError("Could not allocate the EfficientTAM key-mask buffer.")
        }
        return buffer
    }

    public func run(
        imageEmbeddingBuffer: MTLBuffer,
        memoryFeaturesBuffer: MTLBuffer,
        memoryPositionBuffer: MTLBuffer,
        objectPointersBuffer: MTLBuffer? = nil,
        keyMaskBuffer: MTLBuffer? = nil
    ) throws -> [Float]
    {
        try self.validate(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            memoryFeaturesBuffer: memoryFeaturesBuffer,
            memoryPositionBuffer: memoryPositionBuffer,
            objectPointersBuffer: objectPointersBuffer,
            keyMaskBuffer: keyMaskBuffer
        )
        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }
        guard let result = self.executable.run(
            with: self.commandQueue,
            inputs: self.inputs(
                imageEmbeddingBuffer: imageEmbeddingBuffer,
                memoryFeaturesBuffer: memoryFeaturesBuffer,
                memoryPositionBuffer: memoryPositionBuffer,
                objectPointersBuffer: objectPointersBuffer,
                keyMaskBuffer: keyMaskBuffer
            ),
            results: nil,
            executionDescriptor: nil
        ).first else
        {
            throw EfficientTAMError("EfficientTAM memory attention produced no output tensor.")
        }
        return self.read(result, slot: slot)
    }

    @discardableResult
    public func encode(
        imageEmbeddingBuffer: MTLBuffer,
        memoryFeaturesBuffer: MTLBuffer,
        memoryPositionBuffer: MTLBuffer,
        objectPointersBuffer: MTLBuffer? = nil,
        keyMaskBuffer: MTLBuffer? = nil,
        outputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> Bool
    {
        try self.validate(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            memoryFeaturesBuffer: memoryFeaturesBuffer,
            memoryPositionBuffer: memoryPositionBuffer,
            objectPointersBuffer: objectPointersBuffer,
            keyMaskBuffer: keyMaskBuffer,
            outputBuffer: outputBuffer,
            commandBuffer: commandBuffer
        )
        guard let slot = self.acquireSlotNonBlocking() else { return false }
        let output = MPSGraphTensorData(outputBuffer, shape: self.outputTensor.shape ?? [], dataType: .float32)
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        let target = EfficientTAMCommandBuffer.target(for: commandBuffer)
        let mpsCommandBuffer = target.commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: self.inputs(
                    imageEmbeddingBuffer: imageEmbeddingBuffer,
                    memoryFeaturesBuffer: memoryFeaturesBuffer,
                    memoryPositionBuffer: memoryPositionBuffer,
                    objectPointersBuffer: objectPointersBuffer,
                    keyMaskBuffer: keyMaskBuffer
                ),
                results: [output],
                executionDescriptor: descriptor
            )
        }
        do
        {
            try EfficientTAMCommandBuffer.finish(target, commit: commit) { [weak self] in self?.releaseSlot(slot) }
        }
        catch
        {
            self.releaseSlot(slot)
            throw error
        }
        return true
    }

    private func inputs(
        imageEmbeddingBuffer: MTLBuffer,
        memoryFeaturesBuffer: MTLBuffer,
        memoryPositionBuffer: MTLBuffer,
        objectPointersBuffer: MTLBuffer?,
        keyMaskBuffer: MTLBuffer?
    ) -> [MPSGraphTensorData]
    {
        var data: [MPSGraphTensor: MPSGraphTensorData] = [
            self.imageTensor: MPSGraphTensorData(
                imageEmbeddingBuffer, shape: self.imageTensor.shape ?? [], dataType: .float32
            ),
            self.memoryTensor: MPSGraphTensorData(
                memoryFeaturesBuffer, shape: self.memoryTensor.shape ?? [], dataType: .float32
            ),
            self.memoryPositionTensor: MPSGraphTensorData(
                memoryPositionBuffer, shape: self.memoryPositionTensor.shape ?? [], dataType: .float32
            ),
        ]
        if let objectPointersTensor, let objectPointersBuffer
        {
            data[objectPointersTensor] = MPSGraphTensorData(
                objectPointersBuffer, shape: objectPointersTensor.shape ?? [], dataType: .float32
            )
        }
        if let keyMaskTensor, let keyMaskBuffer
        {
            data[keyMaskTensor] = MPSGraphTensorData(
                keyMaskBuffer, shape: keyMaskTensor.shape ?? [], dataType: .float32
            )
        }
        return self.feedOrder.compactMap { data[$0] }
    }

    private func validate(
        imageEmbeddingBuffer: MTLBuffer,
        memoryFeaturesBuffer: MTLBuffer,
        memoryPositionBuffer: MTLBuffer,
        objectPointersBuffer: MTLBuffer?,
        keyMaskBuffer: MTLBuffer?,
        outputBuffer: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws
    {
        guard imageEmbeddingBuffer.length >= self.imageEmbeddingBufferLength else
        {
            throw EfficientTAMError("The current image-embedding buffer is too small.")
        }
        guard memoryFeaturesBuffer.length >= self.memoryFeaturesBufferLength else
        {
            throw EfficientTAMError("The memory-features buffer is too small.")
        }
        guard memoryPositionBuffer.length >= self.memoryPositionBufferLength else
        {
            throw EfficientTAMError("The memory-position buffer is too small.")
        }
        if self.objectPointerCount > 0
        {
            guard let objectPointersBuffer,
                  objectPointersBuffer.length >= self.objectPointersBufferLength else
            {
                throw EfficientTAMError("The object-pointers buffer is missing or too small.")
            }
        }
        if self.usesKeyMask
        {
            let keyMaskLength = (self.memoryFrameCount * 1024 + self.objectPointerCount * 4) * MemoryLayout<Float>.stride
            guard let keyMaskBuffer, keyMaskBuffer.length >= keyMaskLength else
            {
                throw EfficientTAMError("The key-mask buffer is missing or too small.")
            }
        }
        if let outputBuffer, outputBuffer.length < self.outputBufferLength
        {
            throw EfficientTAMError("The memory-attention output buffer is too small.")
        }
        if let commandBuffer, commandBuffer.device !== self.commandQueue.device
        {
            throw EfficientTAMError("The command buffer and EfficientTAM memory attention use different Metal devices.")
        }
    }

    private func read(_ result: MPSGraphTensorData, slot: Int) -> [Float]
    {
        let cache = self.outputCaches[slot]
        cache.values.withUnsafeMutableBufferPointer
        {
            result.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil)
        }
        return cache.values
    }

    private func acquireSlotBlocking() -> Int
    {
        self.slotPool.acquire()
    }

    private func acquireSlotNonBlocking() -> Int?
    {
        self.slotPool.tryAcquire()
    }

    private func releaseSlot(_ slot: Int)
    {
        self.slotPool.release(slot)
    }
}

private enum SelfMemoryShape
{
    static let pixelCount = EfficientTAMMemoryEncoder.memoryWidth * EfficientTAMMemoryEncoder.memoryHeight
}

private struct EfficientTAMMemoryAttentionGraphBuilder
{
    let graph: MPSGraph
    let weights: EfficientTAMWeights
    let memoryFrameCount: Int
    let objectPointerCount: Int

    func build(
        imageEmbedding: MPSGraphTensor,
        memoryFeatures: MPSGraphTensor,
        memoryPosition: MPSGraphTensor,
        objectPointers: MPSGraphTensor?,
        keyMask: MPSGraphTensor?
    ) throws -> MPSGraphTensor
    {
        let noMemory = self.graph.reshape(
            try self.weights.constant(self.graph, named: "no_mem_embed"),
            shape: [1, 256, 1, 1],
            name: nil
        )
        let rawImage = self.graph.subtraction(imageEmbedding, noMemory, name: nil)
        var current = self.graph.reshape(rawImage, shape: [1, 256, 1024], name: nil)
        current = self.graph.transpose(current, permutation: [0, 2, 1], name: nil)
        let currentPosition = self.currentPositionEmbedding()
        current = self.graph.addition(
            current,
            self.graph.multiplication(currentPosition, self.scalar(0.1), name: nil),
            name: nil
        )

        var memory = self.flattenMemory(memoryFeatures)
        var position = self.flattenMemory(memoryPosition)
        if let objectPointers
        {
            let pointerTokens = self.graph.reshape(
                objectPointers,
                shape: [1, (self.objectPointerCount * 4) as NSNumber, 64],
                name: nil
            )
            memory = self.graph.concatTensors([memory, pointerTokens], dimension: 1, name: nil)
            let zeroPosition = self.graph.constant(
                0,
                shape: [1, (self.objectPointerCount * 4) as NSNumber, 64],
                dataType: .float32
            )
            position = self.graph.concatTensors([position, zeroPosition], dimension: 1, name: nil)
        }

        for index in 0..<4
        {
            let prefix = "memory_attention.layers.\(index)"
            var normalized = try self.layerNorm(current, prefix: "\(prefix).norm1")
            let selfAttention = try self.ropeAttention(
                query: normalized,
                key: normalized,
                value: normalized,
                prefix: "\(prefix).self_attn",
                spatialKeyRepeats: 1,
                unrotatedKeyCount: 0,
                keyMask: nil
            )
            current = self.graph.addition(current, selfAttention, name: nil)

            normalized = try self.layerNorm(current, prefix: "\(prefix).norm2")
            let positionedMemory = self.graph.addition(memory, position, name: nil)
            let crossAttention = try self.ropeAttention(
                query: normalized,
                key: positionedMemory,
                value: memory,
                prefix: "\(prefix).cross_attn_image",
                spatialKeyRepeats: self.memoryFrameCount,
                unrotatedKeyCount: self.objectPointerCount * 4,
                keyMask: keyMask
            )
            current = self.graph.addition(current, crossAttention, name: nil)

            normalized = try self.layerNorm(current, prefix: "\(prefix).norm3")
            var feedForward = try self.linear(normalized, prefix: "\(prefix).linear1")
            feedForward = self.graph.reLU(with: feedForward, name: nil)
            feedForward = try self.linear(feedForward, prefix: "\(prefix).linear2")
            current = self.graph.addition(current, feedForward, name: nil)
        }
        current = try self.layerNorm(current, prefix: "memory_attention.norm")
        current = self.graph.transpose(current, permutation: [0, 2, 1], name: nil)
        return self.graph.reshape(current, shape: [1, 256, 32, 32], name: "conditioned_image_embedding")
    }

    private func flattenMemory(_ input: MPSGraphTensor) -> MPSGraphTensor
    {
        var output = self.graph.reshape(
            input,
            shape: [self.memoryFrameCount as NSNumber, 64, 1024],
            name: nil
        )
        output = self.graph.transpose(output, permutation: [0, 2, 1], name: nil)
        return self.graph.reshape(output, shape: [1, (self.memoryFrameCount * 1024) as NSNumber, 64], name: nil)
    }

    private func ropeAttention(
        query: MPSGraphTensor,
        key: MPSGraphTensor,
        value: MPSGraphTensor,
        prefix: String,
        spatialKeyRepeats: Int,
        unrotatedKeyCount: Int,
        keyMask: MPSGraphTensor?
    ) throws -> MPSGraphTensor
    {
        var projectedQuery = try self.linear(query, prefix: "\(prefix).q_proj")
        var projectedKey = try self.linear(key, prefix: "\(prefix).k_proj")
        let projectedValue = try self.linear(value, prefix: "\(prefix).v_proj")
        projectedQuery = self.applyRoPE(projectedQuery, repeats: 1)
        let rotatedKeyCount = spatialKeyRepeats * 1024
        let spatialKey = self.graph.sliceTensor(
            projectedKey,
            dimension: 1,
            start: 0,
            length: rotatedKeyCount,
            name: nil
        )
        let rotatedSpatialKey = self.applyRoPE(spatialKey, repeats: spatialKeyRepeats)
        if unrotatedKeyCount > 0
        {
            let pointerKey = self.graph.sliceTensor(
                projectedKey,
                dimension: 1,
                start: rotatedKeyCount,
                length: unrotatedKeyCount,
                name: nil
            )
            projectedKey = self.graph.concatTensors([rotatedSpatialKey, pointerKey], dimension: 1, name: nil)
        }
        else
        {
            projectedKey = rotatedSpatialKey
        }
        let keyCount = (rotatedKeyCount + unrotatedKeyCount) as NSNumber
        let attended = EfficientTAMAttentionOps.attention(
            graph: self.graph,
            query: self.graph.reshape(projectedQuery, shape: [1, 1, 1024, 256], name: nil),
            key: self.graph.reshape(projectedKey, shape: [1, 1, keyCount, 256], name: nil),
            value: self.graph.reshape(projectedValue, shape: [1, 1, keyCount, 256], name: nil),
            mask: keyMask.map { self.graph.reshape($0, shape: [1, 1, 1, keyCount], name: nil) },
            scale: 1 / sqrt(Float(256))
        )
        return try self.linear(
            self.graph.reshape(attended, shape: [1, 1024, 256], name: nil),
            prefix: "\(prefix).out_proj"
        )
    }

    private func applyRoPE(_ input: MPSGraphTensor, repeats: Int) -> MPSGraphTensor
    {
        let tokenCount = repeats * 1024
        let reshaped = self.graph.reshape(input, shape: [1, tokenCount as NSNumber, 128, 2], name: nil)
        let real = self.graph.sliceTensor(reshaped, dimension: 3, start: 0, length: 1, name: nil)
        let imaginary = self.graph.sliceTensor(reshaped, dimension: 3, start: 1, length: 1, name: nil)
        let frequencies = Self.rotaryFrequencies(repeats: repeats)
        let cosine = self.graph.constant(
            Data(bytes: frequencies.cosine, count: frequencies.cosine.count * MemoryLayout<Float>.stride),
            shape: [1, tokenCount as NSNumber, 128, 1],
            dataType: .float32
        )
        let sine = self.graph.constant(
            Data(bytes: frequencies.sine, count: frequencies.sine.count * MemoryLayout<Float>.stride),
            shape: [1, tokenCount as NSNumber, 128, 1],
            dataType: .float32
        )
        let rotatedReal = self.graph.subtraction(
            self.graph.multiplication(real, cosine, name: nil),
            self.graph.multiplication(imaginary, sine, name: nil),
            name: nil
        )
        let rotatedImaginary = self.graph.addition(
            self.graph.multiplication(real, sine, name: nil),
            self.graph.multiplication(imaginary, cosine, name: nil),
            name: nil
        )
        let paired = self.graph.concatTensors([rotatedReal, rotatedImaginary], dimension: 3, name: nil)
        return self.graph.reshape(paired, shape: [1, tokenCount as NSNumber, 256], name: nil)
    }

    private static func rotaryFrequencies(repeats: Int) -> (cosine: [Float], sine: [Float])
    {
        var baseCosine = [Float](repeating: 0, count: 1024 * 128)
        var baseSine = [Float](repeating: 0, count: 1024 * 128)
        for token in 0..<1024
        {
            let x = Float(token % 32)
            let y = Float(token / 32)
            for pair in 0..<128
            {
                let axisIndex = pair % 64
                let coordinate = pair < 64 ? x : y
                let frequency = 1 / pow(10_000, Float(axisIndex * 4) / 256)
                let angle = coordinate * frequency
                baseCosine[token * 128 + pair] = cos(angle)
                baseSine[token * 128 + pair] = sin(angle)
            }
        }
        return (
            Array(repeating: baseCosine, count: repeats).flatMap { $0 },
            Array(repeating: baseSine, count: repeats).flatMap { $0 }
        )
    }

    private func currentPositionEmbedding() -> MPSGraphTensor
    {
        let values = Self.sinePositionEmbedding(channels: 256)
        return self.graph.constant(
            Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
            shape: [1, 1024, 256],
            dataType: .float32
        )
    }

    private static func sinePositionEmbedding(channels: Int) -> [Float]
    {
        let internalFeatureCount = channels / 2
        let scale = 2 * Float.pi
        var output = [Float](repeating: 0, count: 1024 * channels)
        for y in 0..<32
        {
            let normalizedY = Float(y + 1) / (32 + 1e-6) * scale
            for x in 0..<32
            {
                let normalizedX = Float(x + 1) / (32 + 1e-6) * scale
                let token = y * 32 + x
                for feature in 0..<internalFeatureCount
                {
                    let divisor = pow(10_000, 2 * Float(feature / 2) / Float(internalFeatureCount))
                    output[token * channels + feature] = feature.isMultiple(of: 2)
                        ? sin(normalizedY / divisor) : cos(normalizedY / divisor)
                    output[token * channels + internalFeatureCount + feature] = feature.isMultiple(of: 2)
                        ? sin(normalizedX / divisor) : cos(normalizedX / divisor)
                }
            }
        }
        return output
    }

    private func linear(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let weight = try self.weights.constant(self.graph, named: "\(prefix).weight")
        let transposed = self.graph.transpose(weight, permutation: [1, 0], name: nil)
        let product = self.graph.matrixMultiplication(primary: input, secondary: transposed, name: nil)
        return self.graph.addition(product, try self.weights.constant(self.graph, named: "\(prefix).bias"), name: nil)
    }

    private func layerNorm(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let mean = self.graph.mean(of: input, axes: [-1], name: nil)
        let variance = self.graph.variance(of: input, mean: mean, axes: [-1], name: nil)
        return self.graph.normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: try self.weights.constant(self.graph, named: "\(prefix).weight"),
            beta: try self.weights.constant(self.graph, named: "\(prefix).bias"),
            epsilon: 1e-5,
            name: nil
        )
    }

    private func scalar(_ value: Float) -> MPSGraphTensor
    {
        self.graph.constant(Double(value), dataType: .float32)
    }
}
