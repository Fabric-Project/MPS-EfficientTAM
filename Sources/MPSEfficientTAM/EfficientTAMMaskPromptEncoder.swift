import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Converts one prior 128x128 mask-logit plane into the dense prompt consumed
/// by ``EfficientTAMPromptDecoder``. Keeping this stage independent lets
/// clients cache, replace, or schedule iterative prompting themselves.
public final class EfficientTAMMaskPromptEncoder
{
    public static let inputWidth = 128
    public static let inputHeight = 128
    public static let outputChannels = 256
    public static let outputWidth = 32
    public static let outputHeight = 32

    public var inputBufferLength: Int
    {
        Self.inputWidth * Self.inputHeight * MemoryLayout<Float>.stride
    }

    public var outputBufferLength: Int
    {
        Self.outputChannels * Self.outputWidth * Self.outputHeight * MemoryLayout<Float>.stride
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
        var values = [Float](
            repeating: 0,
            count: EfficientTAMMaskPromptEncoder.outputChannels
                * EfficientTAMMaskPromptEncoder.outputWidth
                * EfficientTAMMaskPromptEncoder.outputHeight
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
        let builder = EfficientTAMMaskPromptGraphBuilder(graph: self.graph, weights: weights)
        let input = self.graph.placeholder(shape: [1, 1, 128, 128], dataType: .float32, name: "mask_logits")
        self.inputTensor = input
        self.outputTensor = try builder.build(input)

        let device = MPSGraphDevice(mtlDevice: commandQueue.device)
        let inputType = MPSGraphShapedType(shape: input.shape ?? [], dataType: .float32)
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level1
        descriptor.waitForCompilationCompletion = true
        self.executable = self.graph.compile(
            with: device,
            feeds: [input: inputType],
            targetTensors: [self.outputTensor],
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        self.executable.specialize(with: device, inputTypes: [inputType], compilationDescriptor: descriptor)
    }

    public func run(maskLogitsBuffer: MTLBuffer) throws -> [Float]
    {
        try self.validate(input: maskLogitsBuffer)
        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }
        let input = MPSGraphTensorData(maskLogitsBuffer, shape: self.inputTensor.shape ?? [], dataType: .float32)
        guard let result = self.executable.run(
            with: self.commandQueue,
            inputs: [input],
            results: nil,
            executionDescriptor: nil
        ).first else
        {
            throw EfficientTAMError("EfficientTAM mask-prompt encoding produced no output tensor.")
        }
        let cache = self.outputCaches[slot]
        cache.values.withUnsafeMutableBufferPointer
        {
            result.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil)
        }
        return cache.values
    }

    @discardableResult
    public func submit(
        maskLogitsBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool,
        completion: @escaping (Result<[Float], any Error>) -> Void
    ) throws -> Bool
    {
        try self.validate(input: maskLogitsBuffer, commandBuffer: commandBuffer)
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
                let cache = self.outputCaches[slot]
                cache.values.withUnsafeMutableBufferPointer
                {
                    result.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil)
                }
                completion(.success(cache.values))
            }
            else
            {
                completion(.failure(EfficientTAMError("EfficientTAM mask-prompt encoding produced no output tensor.")))
            }
        }
        let input = MPSGraphTensorData(maskLogitsBuffer, shape: self.inputTensor.shape ?? [], dataType: .float32)
        let mpsCommandBuffer = EfficientTAMCommandBuffer.target(for: commandBuffer).commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(to: mpsCommandBuffer, inputs: [input], results: nil, executionDescriptor: descriptor)
            if commit { mpsCommandBuffer.commit() }
        }
        return true
    }

    @discardableResult
    public func encode(
        maskLogitsBuffer: MTLBuffer,
        densePromptEmbeddingBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> Bool
    {
        try self.validate(input: maskLogitsBuffer, output: densePromptEmbeddingBuffer, commandBuffer: commandBuffer)
        guard let slot = self.acquireSlotNonBlocking() else { return false }
        let input = MPSGraphTensorData(maskLogitsBuffer, shape: self.inputTensor.shape ?? [], dataType: .float32)
        let output = MPSGraphTensorData(
            densePromptEmbeddingBuffer,
            shape: self.outputTensor.shape ?? [],
            dataType: .float32
        )
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        let target = EfficientTAMCommandBuffer.target(for: commandBuffer)
        let mpsCommandBuffer = target.commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(to: mpsCommandBuffer, inputs: [input], results: [output], executionDescriptor: descriptor)
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

    private func validate(
        input: MTLBuffer,
        output: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws
    {
        guard input.length >= self.inputBufferLength else
        {
            throw EfficientTAMError("The prior mask-logits buffer is too small.")
        }
        if let output, output.length < self.outputBufferLength
        {
            throw EfficientTAMError("The dense-prompt embedding buffer is too small.")
        }
        if let commandBuffer, commandBuffer.device !== self.commandQueue.device
        {
            throw EfficientTAMError("The command buffer and EfficientTAM mask-prompt encoder use different Metal devices.")
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
}

private struct EfficientTAMMaskPromptGraphBuilder
{
    let graph: MPSGraph
    let weights: EfficientTAMWeights

    func build(_ input: MPSGraphTensor) throws -> MPSGraphTensor
    {
        var output = try self.convolution(input, prefix: "sam_prompt_encoder.mask_downscaling.0", stride: 2)
        output = try self.layerNorm2D(output, prefix: "sam_prompt_encoder.mask_downscaling.1")
        output = self.gelu(output)
        output = try self.convolution(output, prefix: "sam_prompt_encoder.mask_downscaling.3", stride: 2)
        output = try self.layerNorm2D(output, prefix: "sam_prompt_encoder.mask_downscaling.4")
        output = self.gelu(output)
        return try self.convolution(output, prefix: "sam_prompt_encoder.mask_downscaling.6", stride: 1)
    }

    private func convolution(_ input: MPSGraphTensor, prefix: String, stride: Int) throws -> MPSGraphTensor
    {
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: 0,
            paddingRight: 0,
            paddingTop: 0,
            paddingBottom: 0,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else
        {
            throw EfficientTAMError("Could not create the EfficientTAM mask-prompt convolution descriptor.")
        }
        var output = self.graph.convolution2D(
            input,
            weights: try self.weights.constant(self.graph, named: "\(prefix).weight"),
            descriptor: descriptor,
            name: prefix
        )
        let bias = try self.weights.floats(named: "\(prefix).bias")
        output = self.graph.addition(output, self.channelConstant(bias), name: nil)
        return output
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
        let inverseSquareRootTwo = self.graph.constant(1 / sqrt(2), dataType: .float32)
        let erf = self.graph.erf(with: self.graph.multiplication(input, inverseSquareRootTwo, name: nil), name: nil)
        let half = self.graph.constant(0.5, dataType: .float32)
        let one = self.graph.constant(1, dataType: .float32)
        return self.graph.multiplication(
            self.graph.multiplication(input, half, name: nil),
            self.graph.addition(one, erf, name: nil),
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
}
