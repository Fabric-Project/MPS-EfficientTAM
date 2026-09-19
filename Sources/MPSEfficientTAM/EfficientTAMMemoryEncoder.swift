import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Encodes a selected mask and its frame embedding into EfficientTAM's spatial
/// video memory. The input mask is one low-resolution `[1, 1, 128, 128]`
/// logit plane from the prompt decoder; official 512-pixel resizing and memory
/// sigmoid scaling are included in this stage.
public final class EfficientTAMMemoryEncoder
{
    public static let memoryChannels = 64
    public static let memoryWidth = 32
    public static let memoryHeight = 32

    public var imageEmbeddingBufferLength: Int
    {
        EfficientTAMImageEncoder.embeddingChannels
            * EfficientTAMImageEncoder.embeddingWidth
            * EfficientTAMImageEncoder.embeddingHeight
            * MemoryLayout<Float>.stride
    }

    public var maskLogitsBufferLength: Int
    {
        EfficientTAMPromptDecoder.maskWidth
            * EfficientTAMPromptDecoder.maskHeight
            * MemoryLayout<Float>.stride
    }

    public var memoryFeaturesBufferLength: Int
    {
        Self.memoryChannels * Self.memoryWidth * Self.memoryHeight * MemoryLayout<Float>.stride
    }

    private let graph = MPSGraph()
    private let commandQueue: MTLCommandQueue
    private let imageEmbeddingTensor: MPSGraphTensor
    private let maskLogitsTensor: MPSGraphTensor
    private let memoryFeaturesTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable
    private let slotPool: EfficientTAMSlotPool
    private let outputCaches: [OutputCache]

    private final class OutputCache
    {
        var values = [Float](
            repeating: 0,
            count: EfficientTAMMemoryEncoder.memoryChannels
                * EfficientTAMMemoryEncoder.memoryWidth
                * EfficientTAMMemoryEncoder.memoryHeight
        )
    }

    public convenience init(commandQueue: MTLCommandQueue, maxFramesInFlight: Int = 3) throws
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
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
    }

    public init(
        weightsBinaryURL: URL,
        weightsManifestURL: URL,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3
    ) throws
    {
        guard maxFramesInFlight > 0 else
        {
            throw EfficientTAMError("EfficientTAM maxFramesInFlight must be positive.")
        }
        self.commandQueue = commandQueue
        self.slotPool = EfficientTAMSlotPool(count: maxFramesInFlight)
        self.outputCaches = (0..<maxFramesInFlight).map { _ in OutputCache() }

        let weights = try EfficientTAMWeights(binaryURL: weightsBinaryURL, manifestURL: weightsManifestURL)
        let builder = EfficientTAMMemoryEncoderGraphBuilder(graph: self.graph, weights: weights)
        let imageEmbedding = self.graph.placeholder(
            shape: [1, 256, 32, 32],
            dataType: .float32,
            name: "image_embedding"
        )
        let maskLogits = self.graph.placeholder(
            shape: [1, 1, 128, 128],
            dataType: .float32,
            name: "selected_mask_logits"
        )
        self.imageEmbeddingTensor = imageEmbedding
        self.maskLogitsTensor = maskLogits
        self.memoryFeaturesTensor = try builder.build(
            imageEmbedding: imageEmbedding,
            selectedMaskLogits: maskLogits
        )

        let device = MPSGraphDevice(mtlDevice: commandQueue.device)
        let imageType = MPSGraphShapedType(shape: imageEmbedding.shape ?? [], dataType: .float32)
        let maskType = MPSGraphShapedType(shape: maskLogits.shape ?? [], dataType: .float32)
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level1
        descriptor.waitForCompilationCompletion = true
        self.executable = self.graph.compile(
            with: device,
            feeds: [imageEmbedding: imageType, maskLogits: maskType],
            targetTensors: [self.memoryFeaturesTensor],
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        self.executable.specialize(
            with: device,
            inputTypes: [imageType, maskType],
            compilationDescriptor: descriptor
        )
    }

    public func run(imageEmbeddingBuffer: MTLBuffer, maskLogitsBuffer: MTLBuffer) throws -> [Float]
    {
        try self.validate(imageEmbeddingBuffer: imageEmbeddingBuffer, maskLogitsBuffer: maskLogitsBuffer)
        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }
        guard let result = self.executable.run(
            with: self.commandQueue,
            inputs: self.inputs(imageEmbeddingBuffer: imageEmbeddingBuffer, maskLogitsBuffer: maskLogitsBuffer),
            results: nil,
            executionDescriptor: nil
        ).first else
        {
            throw EfficientTAMError("EfficientTAM memory encoding produced no output tensor.")
        }
        return self.read(result, slot: slot)
    }

    @discardableResult
    public func submit(
        imageEmbeddingBuffer: MTLBuffer,
        maskLogitsBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool,
        completion: @escaping (Result<[Float], any Error>) -> Void
    ) throws -> Bool
    {
        try self.validate(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            maskLogitsBuffer: maskLogitsBuffer,
            commandBuffer: commandBuffer
        )
        guard let slot = self.acquireSlotNonBlocking() else { return false }
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        descriptor.completionHandler = { [weak self] results, error in
            guard let self else { return }
            defer { self.releaseSlot(slot) }
            if let error
            {
                completion(.failure(error))
            }
            else if let result = results.first
            {
                completion(.success(self.read(result, slot: slot)))
            }
            else
            {
                completion(.failure(EfficientTAMError("EfficientTAM memory encoding produced no output tensor.")))
            }
        }
        let mpsCommandBuffer = EfficientTAMCommandBuffer.target(for: commandBuffer).commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: self.inputs(imageEmbeddingBuffer: imageEmbeddingBuffer, maskLogitsBuffer: maskLogitsBuffer),
                results: nil,
                executionDescriptor: descriptor
            )
            if commit { mpsCommandBuffer.commit() }
        }
        return true
    }

    @discardableResult
    public func encode(
        imageEmbeddingBuffer: MTLBuffer,
        maskLogitsBuffer: MTLBuffer,
        memoryFeaturesBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> Bool
    {
        try self.validate(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            maskLogitsBuffer: maskLogitsBuffer,
            memoryFeaturesBuffer: memoryFeaturesBuffer,
            commandBuffer: commandBuffer
        )
        guard let slot = self.acquireSlotNonBlocking() else { return false }
        let output = MPSGraphTensorData(
            memoryFeaturesBuffer,
            shape: self.memoryFeaturesTensor.shape ?? [],
            dataType: .float32
        )
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        let target = EfficientTAMCommandBuffer.target(for: commandBuffer)
        let mpsCommandBuffer = target.commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: self.inputs(imageEmbeddingBuffer: imageEmbeddingBuffer, maskLogitsBuffer: maskLogitsBuffer),
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

    /// Spatial memory position encoding is constant for the fixed 32x32 grid.
    public static func positionEmbedding() -> [Float]
    {
        let internalFeatureCount = Self.memoryChannels / 2
        let scale = 2 * Float.pi
        var output = [Float](repeating: 0, count: Self.memoryChannels * Self.memoryWidth * Self.memoryHeight)
        for y in 0..<Self.memoryHeight
        {
            let normalizedY = Float(y + 1) / (Float(Self.memoryHeight) + 1e-6) * scale
            for x in 0..<Self.memoryWidth
            {
                let normalizedX = Float(x + 1) / (Float(Self.memoryWidth) + 1e-6) * scale
                for feature in 0..<internalFeatureCount
                {
                    let divisor = pow(10_000, 2 * Float(feature / 2) / Float(internalFeatureCount))
                    let yValue = feature.isMultiple(of: 2)
                        ? sin(normalizedY / divisor)
                        : cos(normalizedY / divisor)
                    let xValue = feature.isMultiple(of: 2)
                        ? sin(normalizedX / divisor)
                        : cos(normalizedX / divisor)
                    let pixelIndex = y * Self.memoryWidth + x
                    output[feature * Self.memoryWidth * Self.memoryHeight + pixelIndex] = yValue
                    output[(internalFeatureCount + feature) * Self.memoryWidth * Self.memoryHeight + pixelIndex] = xValue
                }
            }
        }
        return output
    }

    public func makePositionEmbeddingBuffer() throws -> MTLBuffer
    {
        let values = Self.positionEmbedding()
        guard let buffer = self.commandQueue.device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride
        ) else
        {
            throw EfficientTAMError("Could not allocate the EfficientTAM memory-position buffer.")
        }
        return buffer
    }

    private func inputs(imageEmbeddingBuffer: MTLBuffer, maskLogitsBuffer: MTLBuffer) -> [MPSGraphTensorData]
    {
        [
            MPSGraphTensorData(
                imageEmbeddingBuffer,
                shape: self.imageEmbeddingTensor.shape ?? [],
                dataType: .float32
            ),
            MPSGraphTensorData(maskLogitsBuffer, shape: self.maskLogitsTensor.shape ?? [], dataType: .float32),
        ]
    }

    private func validate(
        imageEmbeddingBuffer: MTLBuffer,
        maskLogitsBuffer: MTLBuffer,
        memoryFeaturesBuffer: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws
    {
        guard imageEmbeddingBuffer.length >= self.imageEmbeddingBufferLength else
        {
            throw EfficientTAMError("The image-embedding buffer is too small for memory encoding.")
        }
        guard maskLogitsBuffer.length >= self.maskLogitsBufferLength else
        {
            throw EfficientTAMError("The selected mask-logits buffer is too small for memory encoding.")
        }
        if let memoryFeaturesBuffer, memoryFeaturesBuffer.length < self.memoryFeaturesBufferLength
        {
            throw EfficientTAMError("The memory-features buffer is too small.")
        }
        if let commandBuffer, commandBuffer.device !== self.commandQueue.device
        {
            throw EfficientTAMError("The command buffer and EfficientTAM memory encoder use different Metal devices.")
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

private struct EfficientTAMMemoryEncoderGraphBuilder
{
    let graph: MPSGraph
    let weights: EfficientTAMWeights

    func build(imageEmbedding: MPSGraphTensor, selectedMaskLogits: MPSGraphTensor) throws -> MPSGraphTensor
    {
        let noMemory = self.graph.reshape(
            try self.weights.constant(self.graph, named: "no_mem_embed"),
            shape: [1, 256, 1, 1],
            name: nil
        )
        let rawImageFeature = self.graph.subtraction(imageEmbedding, noMemory, name: nil)
        var mask = self.graph.resize(
            selectedMaskLogits,
            size: [512, 512],
            mode: .bilinear,
            centerResult: true,
            alignCorners: false,
            layout: .NCHW,
            name: nil
        )
        mask = self.graph.sigmoid(with: mask, name: nil)
        mask = self.graph.addition(
            self.graph.multiplication(mask, self.scalar(20), name: nil),
            self.scalar(-10),
            name: nil
        )

        let convolutionIndexes = [0, 3, 6, 9]
        let normalizationIndexes = [1, 4, 7, 10]
        for stage in 0..<4
        {
            mask = try self.convolution(
                mask,
                prefix: "memory_encoder.mask_downsampler.encoder.\(convolutionIndexes[stage])",
                stride: 2,
                padding: 1
            )
            mask = try self.layerNorm2D(
                mask,
                prefix: "memory_encoder.mask_downsampler.encoder.\(normalizationIndexes[stage])"
            )
            mask = self.gelu(mask)
        }
        mask = try self.convolution(mask, prefix: "memory_encoder.mask_downsampler.encoder.12")

        var output = try self.convolution(rawImageFeature, prefix: "memory_encoder.pix_feat_proj")
        output = self.graph.addition(output, mask, name: nil)
        for index in 0..<2
        {
            output = try self.convNextBlock(output, index: index)
        }
        return try self.convolution(output, prefix: "memory_encoder.out_proj")
    }

    private func convNextBlock(_ input: MPSGraphTensor, index: Int) throws -> MPSGraphTensor
    {
        let prefix = "memory_encoder.fuser.layers.\(index)"
        var output = try self.convolution(input, prefix: "\(prefix).dwconv", padding: 3, groups: 256)
        output = try self.layerNorm2D(output, prefix: "\(prefix).norm")
        output = self.graph.transpose(output, permutation: [0, 2, 3, 1], name: nil)
        output = try self.linear(output, prefix: "\(prefix).pwconv1")
        output = self.gelu(output)
        output = try self.linear(output, prefix: "\(prefix).pwconv2")
        output = self.graph.multiplication(
            output,
            try self.weights.constant(self.graph, named: "\(prefix).gamma"),
            name: nil
        )
        output = self.graph.transpose(output, permutation: [0, 3, 1, 2], name: nil)
        return self.graph.addition(input, output, name: nil)
    }

    private func convolution(
        _ input: MPSGraphTensor,
        prefix: String,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1
    ) throws -> MPSGraphTensor
    {
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: groups,
            paddingLeft: padding,
            paddingRight: padding,
            paddingTop: padding,
            paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else
        {
            throw EfficientTAMError("Could not create EfficientTAM memory convolution descriptor for '\(prefix)'.")
        }
        var output = self.graph.convolution2D(
            input,
            weights: try self.weights.constant(self.graph, named: "\(prefix).weight"),
            descriptor: descriptor,
            name: nil
        )
        output = self.graph.addition(
            output,
            self.channelConstant(try self.weights.floats(named: "\(prefix).bias")),
            name: nil
        )
        return output
    }

    private func linear(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let weight = try self.weights.constant(self.graph, named: "\(prefix).weight")
        let transposed = self.graph.transpose(weight, permutation: [1, 0], name: nil)
        let output = self.graph.matrixMultiplication(primary: input, secondary: transposed, name: nil)
        return self.graph.addition(output, try self.weights.constant(self.graph, named: "\(prefix).bias"), name: nil)
    }

    private func layerNorm2D(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let mean = self.graph.mean(of: input, axes: [1], name: nil)
        let variance = self.graph.variance(of: input, mean: mean, axes: [1], name: nil)
        return self.graph.normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: self.channelConstant(try self.weights.floats(named: "\(prefix).weight")),
            beta: self.channelConstant(try self.weights.floats(named: "\(prefix).bias")),
            epsilon: 1e-6,
            name: nil
        )
    }

    private func gelu(_ input: MPSGraphTensor) -> MPSGraphTensor
    {
        let erf = self.graph.erf(
            with: self.graph.multiplication(input, self.scalar(1 / sqrt(Float(2))), name: nil),
            name: nil
        )
        return self.graph.multiplication(
            self.graph.multiplication(input, self.scalar(0.5), name: nil),
            self.graph.addition(self.scalar(1), erf, name: nil),
            name: nil
        )
    }

    private func channelConstant(_ values: [Float]) -> MPSGraphTensor
    {
        self.graph.constant(
            Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
            shape: [1, values.count as NSNumber, 1, 1],
            dataType: .float32
        )
    }

    private func scalar(_ value: Float) -> MPSGraphTensor
    {
        self.graph.constant(Double(value), dataType: .float32)
    }
}
