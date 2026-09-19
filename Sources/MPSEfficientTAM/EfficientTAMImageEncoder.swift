import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// EfficientTAM-Tiny's official 512x512 image encoder, implemented directly
/// with MPSGraph.
///
/// Input is one tightly packed NHWC float32 RGB image in the 0...1 range.
/// Output is one NCHW `[1, 256, 32, 32]` image embedding. The encoder includes
/// the upstream ImageNet normalization and the learned `no_mem_embed` addition.
/// The output can be cached and reused for any number of prompt-decoder calls.
public final class EfficientTAMImageEncoder
{
    public static let inputWidth = 512
    public static let inputHeight = 512
    public static let embeddingChannels = 256
    public static let embeddingWidth = 32
    public static let embeddingHeight = 32

    public var inputBufferLength: Int
    {
        Self.inputWidth * Self.inputHeight * 3 * MemoryLayout<Float>.stride
    }

    public var outputBufferLength: Int
    {
        Self.embeddingChannels * Self.embeddingWidth * Self.embeddingHeight * MemoryLayout<Float>.stride
    }

    private let graph = MPSGraph()
    private let commandQueue: MTLCommandQueue
    private let inputTensor: MPSGraphTensor
    private let outputTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable
    private let slotPool: EfficientTAMSlotPool
    private let outputCaches: [OutputCache]

    private final class OutputCache
    {
        var values: [Float] = []
    }

    public convenience init(
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
        let builder = EfficientTAMImageEncoderGraphBuilder(graph: self.graph, weights: weights)
        let input = self.graph.placeholder(
            shape: [1, Self.inputHeight as NSNumber, Self.inputWidth as NSNumber, 3],
            dataType: .float32,
            name: "rgb"
        )
        self.inputTensor = input
        self.outputTensor = try builder.build(inputNHWC: input)

        let device = MPSGraphDevice(mtlDevice: commandQueue.device)
        let inputType = MPSGraphShapedType(shape: input.shape ?? [], dataType: .float32)
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level1
        descriptor.waitForCompilationCompletion = true
        // Keep the default full-precision transformer intermediates. In
        // particular, do not opt into `allowFP16Intermediates`: upstream has
        // documented significant image-encoder accuracy loss under FP16.
        self.executable = self.graph.compile(
            with: device,
            feeds: [input: inputType],
            targetTensors: [self.outputTensor],
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        self.executable.specialize(
            with: device,
            inputTypes: [inputType],
            compilationDescriptor: descriptor
        )
    }

    /// Synchronous convenience entry point. This API necessarily waits for
    /// its returned CPU array; real-time callers should use `encode`.
    public func run(inputBuffer: MTLBuffer) throws -> [Float]
    {
        try self.validate(inputBuffer: inputBuffer)
        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }

        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: self.inputTensor.shape ?? [],
            dataType: .float32
        )
        guard let result = self.executable.run(
            with: self.commandQueue,
            inputs: [inputData],
            results: nil,
            executionDescriptor: nil
        ).first else
        {
            throw EfficientTAMError("EfficientTAM image encoding produced no output tensor.")
        }
        return self.floatArray(from: result, slot: slot)
    }

    /// Asynchronously returns a CPU array after GPU completion. Returns false
    /// under normal in-flight backpressure instead of blocking the caller.
    @discardableResult
    public func submit(
        inputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool,
        completion: @escaping (Result<[Float], any Error>) -> Void
    ) throws -> Bool
    {
        try self.validate(inputBuffer: inputBuffer, commandBuffer: commandBuffer)
        guard let slot = self.acquireSlotNonBlocking() else { return false }

        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = false
        executionDescriptor.completionHandler = { [weak self] results, error in
            guard let self else { return }
            defer { self.releaseSlot(slot) }
            if let error
            {
                completion(.failure(error))
            }
            else if let result = results.first
            {
                completion(.success(self.floatArray(from: result, slot: slot)))
            }
            else
            {
                completion(.failure(EfficientTAMError("EfficientTAM image encoding produced no output tensor.")))
            }
        }

        let inputData = MPSGraphTensorData(inputBuffer, shape: self.inputTensor.shape ?? [], dataType: .float32)
        let mpsCommandBuffer = EfficientTAMCommandBuffer.target(for: commandBuffer).commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: [inputData],
                results: nil,
                executionDescriptor: executionDescriptor
            )
            if commit
            {
                mpsCommandBuffer.commit()
            }
        }
        return true
    }

    /// Encodes the complete image encoder without CPU readback or GPU wait.
    /// The output is NCHW `[1, 256, 32, 32]` float32.
    @discardableResult
    public func encode(
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> Bool
    {
        try self.validate(inputBuffer: inputBuffer, commandBuffer: commandBuffer)
        guard outputBuffer.length >= self.outputBufferLength else
        {
            throw EfficientTAMError(
                "Output buffer has \(outputBuffer.length) bytes; EfficientTAM requires \(self.outputBufferLength)."
            )
        }
        guard let slot = self.acquireSlotNonBlocking() else { return false }

        let inputData = MPSGraphTensorData(inputBuffer, shape: self.inputTensor.shape ?? [], dataType: .float32)
        let outputData = MPSGraphTensorData(outputBuffer, shape: self.outputTensor.shape ?? [], dataType: .float32)
        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = false
        let target = EfficientTAMCommandBuffer.target(for: commandBuffer)
        let mpsCommandBuffer = target.commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: [inputData],
                results: [outputData],
                executionDescriptor: executionDescriptor
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

    public func prediction(rgb: [Float]) throws -> [Float]
    {
        let requiredCount = Self.inputWidth * Self.inputHeight * 3
        guard rgb.count == requiredCount else
        {
            throw EfficientTAMError("RGB array has \(rgb.count) values; EfficientTAM requires \(requiredCount).")
        }
        guard let inputBuffer = self.commandQueue.device.makeBuffer(
            bytes: rgb,
            length: rgb.count * MemoryLayout<Float>.stride
        ) else
        {
            throw EfficientTAMError("Could not allocate the EfficientTAM input buffer.")
        }
        return try self.run(inputBuffer: inputBuffer)
    }

    private func validate(inputBuffer: MTLBuffer, commandBuffer: MTLCommandBuffer? = nil) throws
    {
        guard inputBuffer.length >= self.inputBufferLength else
        {
            throw EfficientTAMError(
                "Input buffer has \(inputBuffer.length) bytes; EfficientTAM requires \(self.inputBufferLength)."
            )
        }
        if let commandBuffer, commandBuffer.device !== self.commandQueue.device
        {
            throw EfficientTAMError("The command buffer and EfficientTAM encoder use different Metal devices.")
        }
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

    private func floatArray(from tensorData: MPSGraphTensorData, slot: Int) -> [Float]
    {
        let count = Self.embeddingChannels * Self.embeddingWidth * Self.embeddingHeight
        let cache = self.outputCaches[slot]
        if cache.values.count != count
        {
            cache.values = [Float](repeating: 0, count: count)
        }
        cache.values.withUnsafeMutableBufferPointer { buffer in
            tensorData.mpsndarray().readBytes(buffer.baseAddress!, strideBytes: nil)
        }
        return cache.values
    }
}

private struct EfficientTAMImageEncoderGraphBuilder
{
    let graph: MPSGraph
    let weights: EfficientTAMWeights

    private let grid = 32
    private let embedDimension = 192
    private let numberOfHeads = 3
    private let windowSize = 14
    private let windowBlockIndexes: Set<Int> = [0, 1, 3, 4, 6, 7, 9, 10]

    func build(inputNHWC: MPSGraphTensor) throws -> MPSGraphTensor
    {
        var input = self.graph.transpose(inputNHWC, permutation: [0, 3, 1, 2], name: "input_nchw")
        input = self.graph.division(
            self.graph.subtraction(input, self.channelConstant([0.485, 0.456, 0.406]), name: nil),
            self.channelConstant([0.229, 0.224, 0.225]),
            name: "normalized_input"
        )

        var tokens = try self.convolution(
            input,
            weight: "image_encoder.trunk.patch_embed.proj.weight",
            bias: "image_encoder.trunk.patch_embed.proj.bias",
            stride: 16
        )
        tokens = self.graph.transpose(tokens, permutation: [0, 2, 3, 1], name: nil)
        tokens = self.graph.addition(tokens, try self.positionalEmbedding(), name: "tokens_with_position")

        for index in 0..<12
        {
            tokens = try self.transformerBlock(tokens, index: index, windowed: self.windowBlockIndexes.contains(index))
        }

        var feature = self.graph.transpose(tokens, permutation: [0, 3, 1, 2], name: nil)
        feature = try self.convolution(feature, weight: "image_encoder.neck.convs.0.conv_1x1.weight")
        feature = try self.layerNorm2D(feature, prefix: "image_encoder.neck.convs.0.norm_0")
        feature = try self.convolution(feature, weight: "image_encoder.neck.convs.0.conv_3x3.weight", padding: 1)
        feature = try self.layerNorm2D(feature, prefix: "image_encoder.neck.convs.0.norm_1")
        let noMemoryEmbedding = try self.weights.constant(self.graph, named: "no_mem_embed")
        let broadcastNoMemory = self.graph.reshape(noMemoryEmbedding, shape: [1, 256, 1, 1], name: nil)
        return self.graph.addition(feature, broadcastNoMemory, name: "image_embedding")
    }

    private func transformerBlock(
        _ input: MPSGraphTensor,
        index: Int,
        windowed: Bool
    ) throws -> MPSGraphTensor
    {
        let prefix = "image_encoder.trunk.blocks.\(index)"
        let normalizedAttentionInput = try self.layerNorm(input, prefix: "\(prefix).norm1")
        let attentionInput = windowed ? self.partitionWindows(normalizedAttentionInput) : normalizedAttentionInput
        var attentionOutput = try self.attention(attentionInput, prefix: "\(prefix).attn")
        if windowed
        {
            attentionOutput = self.unpartitionWindows(attentionOutput)
        }
        var output = self.graph.addition(input, attentionOutput, name: nil)
        var mlp = try self.layerNorm(output, prefix: "\(prefix).norm2")
        mlp = try self.linear(mlp, prefix: "\(prefix).mlp.layers.0")
        mlp = self.gelu(mlp)
        mlp = try self.linear(mlp, prefix: "\(prefix).mlp.layers.1")
        output = self.graph.addition(output, mlp, name: "block_\(index)")
        return output
    }

    private func attention(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let shape = input.shape?.map(\.intValue) ?? []
        guard shape.count == 4 else
        {
            throw EfficientTAMError("Unexpected EfficientTAM attention input shape.")
        }
        let batch = shape[0]
        let height = shape[1]
        let width = shape[2]
        let tokenCount = height * width
        let headDimension = self.embedDimension / self.numberOfHeads

        let flat = self.graph.reshape(
            input,
            shape: [batch as NSNumber, tokenCount as NSNumber, self.embedDimension as NSNumber],
            name: nil
        )
        var qkv = try self.linear(flat, prefix: "\(prefix).qkv")
        qkv = self.graph.reshape(
            qkv,
            shape: [batch as NSNumber, tokenCount as NSNumber, 3, self.numberOfHeads as NSNumber, headDimension as NSNumber],
            name: nil
        )
        qkv = self.graph.transpose(qkv, permutation: [2, 0, 3, 1, 4], name: nil)
        let query = self.graph.reshape(
            self.graph.sliceTensor(qkv, dimension: 0, start: 0, length: 1, name: nil),
            shape: [batch as NSNumber, self.numberOfHeads as NSNumber, tokenCount as NSNumber, headDimension as NSNumber],
            name: nil
        )
        let key = self.graph.reshape(
            self.graph.sliceTensor(qkv, dimension: 0, start: 1, length: 1, name: nil),
            shape: [batch as NSNumber, self.numberOfHeads as NSNumber, tokenCount as NSNumber, headDimension as NSNumber],
            name: nil
        )
        let value = self.graph.reshape(
            self.graph.sliceTensor(qkv, dimension: 0, start: 2, length: 1, name: nil),
            shape: [batch as NSNumber, self.numberOfHeads as NSNumber, tokenCount as NSNumber, headDimension as NSNumber],
            name: nil
        )
        var attended = EfficientTAMAttentionOps.attention(
            graph: self.graph,
            query: query,
            key: key,
            value: value,
            mask: nil,
            scale: 1 / sqrt(Float(headDimension))
        )
        attended = self.graph.transpose(attended, permutation: [0, 2, 1, 3], name: nil)
        attended = self.graph.reshape(
            attended,
            shape: [batch as NSNumber, height as NSNumber, width as NSNumber, self.embedDimension as NSNumber],
            name: nil
        )
        return try self.linear(attended, prefix: "\(prefix).proj")
    }

    private func partitionWindows(_ input: MPSGraphTensor) -> MPSGraphTensor
    {
        let padded = self.graph.padTensor(
            input,
            with: .constant,
            leftPadding: [0, 0, 0, 0],
            rightPadding: [0, 10, 10, 0],
            constantValue: 0,
            name: nil
        )
        var windows = self.graph.reshape(padded, shape: [1, 3, 14, 3, 14, 192], name: nil)
        windows = self.graph.transpose(windows, permutation: [0, 1, 3, 2, 4, 5], name: nil)
        return self.graph.reshape(windows, shape: [9, 14, 14, 192], name: nil)
    }

    private func unpartitionWindows(_ input: MPSGraphTensor) -> MPSGraphTensor
    {
        var tokens = self.graph.reshape(input, shape: [1, 3, 3, 14, 14, 192], name: nil)
        tokens = self.graph.transpose(tokens, permutation: [0, 1, 3, 2, 4, 5], name: nil)
        tokens = self.graph.reshape(tokens, shape: [1, 42, 42, 192], name: nil)
        return self.graph.sliceTensor(
            tokens,
            starts: [0, 0, 0, 0],
            ends: [1, 32, 32, 192],
            strides: [1, 1, 1, 1],
            name: nil
        )
    }

    private func linear(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let weight = try self.weights.constant(self.graph, named: "\(prefix).weight")
        let transposedWeight = self.graph.transpose(weight, permutation: [1, 0], name: nil)
        let product = self.graph.matrixMultiplication(primary: input, secondary: transposedWeight, name: nil)
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
            epsilon: 1e-6,
            name: nil
        )
    }

    private func layerNorm2D(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let mean = self.graph.mean(of: input, axes: [1], name: nil)
        let variance = self.graph.variance(of: input, mean: mean, axes: [1], name: nil)
        let gamma = self.channelConstant(try self.weights.floats(named: "\(prefix).weight"))
        let beta = self.channelConstant(try self.weights.floats(named: "\(prefix).bias"))
        return self.graph.normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: gamma,
            beta: beta,
            epsilon: 1e-6,
            name: nil
        )
    }

    private func gelu(_ input: MPSGraphTensor) -> MPSGraphTensor
    {
        let erf = self.graph.erf(with: self.graph.multiplication(input, self.scalar(1 / sqrt(2)), name: nil), name: nil)
        return self.graph.multiplication(
            self.graph.multiplication(input, self.scalar(0.5), name: nil),
            self.graph.addition(self.scalar(1), erf, name: nil),
            name: nil
        )
    }

    private func convolution(
        _ input: MPSGraphTensor,
        weight weightName: String,
        bias biasName: String? = nil,
        stride: Int = 1,
        padding: Int = 0
    ) throws -> MPSGraphTensor
    {
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: padding,
            paddingRight: padding,
            paddingTop: padding,
            paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else
        {
            throw EfficientTAMError("Could not create convolution descriptor for '\(weightName)'.")
        }
        var output = self.graph.convolution2D(
            input,
            weights: try self.weights.constant(self.graph, named: weightName),
            descriptor: descriptor,
            name: weightName
        )
        if let biasName
        {
            output = self.graph.addition(
                output,
                self.channelConstant(try self.weights.floats(named: biasName)),
                name: nil
            )
        }
        return output
    }

    private func positionalEmbedding() throws -> MPSGraphTensor
    {
        let stored = try self.weights.floats(named: "image_encoder.trunk.pos_embed")
        let source = Array(stored.dropFirst(self.embedDimension))
        let resized = Self.bilinearResize(
            source,
            sourceHeight: 14,
            sourceWidth: 14,
            channels: self.embedDimension,
            destinationHeight: self.grid,
            destinationWidth: self.grid
        )
        return self.graph.constant(
            Data(bytes: resized, count: resized.count * MemoryLayout<Float>.stride),
            shape: [1, self.grid as NSNumber, self.grid as NSNumber, self.embedDimension as NSNumber],
            dataType: .float32
        )
    }

    private static func bilinearResize(
        _ source: [Float],
        sourceHeight: Int,
        sourceWidth: Int,
        channels: Int,
        destinationHeight: Int,
        destinationWidth: Int
    ) -> [Float]
    {
        var output = [Float](repeating: 0, count: destinationHeight * destinationWidth * channels)
        for destinationY in 0..<destinationHeight
        {
            let sourceY = (Float(destinationY) + 0.5) * Float(sourceHeight) / Float(destinationHeight) - 0.5
            let y0 = max(0, min(sourceHeight - 1, Int(floor(sourceY))))
            let y1 = max(0, min(sourceHeight - 1, y0 + 1))
            let yFraction = max(0, min(1, sourceY - Float(y0)))
            for destinationX in 0..<destinationWidth
            {
                let sourceX = (Float(destinationX) + 0.5) * Float(sourceWidth) / Float(destinationWidth) - 0.5
                let x0 = max(0, min(sourceWidth - 1, Int(floor(sourceX))))
                let x1 = max(0, min(sourceWidth - 1, x0 + 1))
                let xFraction = max(0, min(1, sourceX - Float(x0)))
                for channel in 0..<channels
                {
                    let topLeft = source[(y0 * sourceWidth + x0) * channels + channel]
                    let topRight = source[(y0 * sourceWidth + x1) * channels + channel]
                    let bottomLeft = source[(y1 * sourceWidth + x0) * channels + channel]
                    let bottomRight = source[(y1 * sourceWidth + x1) * channels + channel]
                    let top = topLeft + (topRight - topLeft) * xFraction
                    let bottom = bottomLeft + (bottomRight - bottomLeft) * xFraction
                    output[(destinationY * destinationWidth + destinationX) * channels + channel] =
                        top + (bottom - top) * yFraction
                }
            }
        }
        return output
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
