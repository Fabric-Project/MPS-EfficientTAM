import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public enum EfficientTAMPromptLabel: Float, Sendable
{
    case padding = -1
    case negativePoint = 0
    case positivePoint = 1
    case topLeftBoxCorner = 2
    case bottomRightBoxCorner = 3
}

public struct EfficientTAMPrompt: Sendable
{
    public let x: Float
    public let y: Float
    public let label: EfficientTAMPromptLabel

    public init(x: Float, y: Float, label: EfficientTAMPromptLabel)
    {
        self.x = x
        self.y = y
        self.label = label
    }
}

public struct EfficientTAMMaskPrediction: Sendable
{
    /// Three NCHW-compatible 128x128 low-resolution mask-logit planes.
    public let maskLogits: [Float]
    /// The model's three sigmoid IoU-quality estimates.
    public let iouPredictions: [Float]
    /// Raw object-presence logit used by EfficientTAM's occlusion handling.
    public let objectScoreLogit: Float
    /// Three candidate object pointers aligned with the three mask candidates.
    public let objectPointers: [Float]
}

/// EfficientTAM's point/box prompt encoder and SAM-style two-way mask
/// decoder, implemented directly with MPSGraph.
///
/// Instances compile for an exact prompt-token count. This preserves the
/// official model's attention semantics without padding every call to an
/// arbitrary maximum. For point-only prompting, append one `.padding` token
/// just as upstream SAM does. A box naturally uses two corner tokens and does
/// not need an additional padding token.
public final class EfficientTAMPromptDecoder
{
    public static let maskCount = 3
    public static let maskWidth = 128
    public static let maskHeight = 128
    public static let objectPointerWidth = 256

    public let promptCount: Int

    public var imageEmbeddingBufferLength: Int
    {
        EfficientTAMImageEncoder.embeddingChannels
            * EfficientTAMImageEncoder.embeddingWidth
            * EfficientTAMImageEncoder.embeddingHeight
            * MemoryLayout<Float>.stride
    }

    public var promptCoordinatesBufferLength: Int
    {
        self.promptCount * 2 * MemoryLayout<Float>.stride
    }

    public var promptLabelsBufferLength: Int
    {
        self.promptCount * MemoryLayout<Float>.stride
    }

    /// One NCHW `[1, 256, 32, 32]` dense prompt embedding. Supplying this
    /// optional input enables iterative refinement from a prior mask.
    public var densePromptEmbeddingBufferLength: Int
    {
        EfficientTAMImageEncoder.embeddingChannels
            * EfficientTAMImageEncoder.embeddingWidth
            * EfficientTAMImageEncoder.embeddingHeight
            * MemoryLayout<Float>.stride
    }

    public var maskLogitsBufferLength: Int
    {
        Self.maskCount * Self.maskWidth * Self.maskHeight * MemoryLayout<Float>.stride
    }

    public var iouPredictionsBufferLength: Int
    {
        Self.maskCount * MemoryLayout<Float>.stride
    }

    public var objectScoreLogitBufferLength: Int
    {
        MemoryLayout<Float>.stride
    }

    public var objectPointersBufferLength: Int
    {
        Self.maskCount * Self.objectPointerWidth * MemoryLayout<Float>.stride
    }

    private let graph = MPSGraph()
    private let commandQueue: MTLCommandQueue
    private let imageEmbeddingTensor: MPSGraphTensor
    private let promptCoordinatesTensor: MPSGraphTensor
    private let promptLabelsTensor: MPSGraphTensor
    private let densePromptEmbeddingTensor: MPSGraphTensor
    private let noMaskEmbeddingBuffer: MTLBuffer
    private let maskLogitsTensor: MPSGraphTensor
    private let iouPredictionsTensor: MPSGraphTensor
    private let objectScoreLogitTensor: MPSGraphTensor
    private let objectPointersTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable
    private let slotPool: EfficientTAMSlotPool
    private let outputCaches: [OutputCache]
    private let trackingOutputBuffers: [TrackingOutputBuffers]

    private final class OutputCache
    {
        var maskLogits: [Float] = []
        var iouPredictions: [Float] = []
        var objectPointers: [Float] = []
    }

    private struct TrackingOutputBuffers
    {
        let objectScoreLogit: MTLBuffer
        let objectPointers: MTLBuffer
    }

    public convenience init(
        promptCount: Int = 2,
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
            promptCount: promptCount,
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
    }

    public init(
        weightsBinaryURL: URL,
        weightsManifestURL: URL,
        promptCount: Int,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3
    ) throws
    {
        guard promptCount > 0 else
        {
            throw EfficientTAMError("EfficientTAM promptCount must be positive.")
        }
        guard maxFramesInFlight > 0 else
        {
            throw EfficientTAMError("EfficientTAM maxFramesInFlight must be positive.")
        }

        self.promptCount = promptCount
        self.commandQueue = commandQueue
        self.slotPool = EfficientTAMSlotPool(count: maxFramesInFlight)
        self.outputCaches = (0..<maxFramesInFlight).map { _ in OutputCache() }
        var trackingOutputBuffers: [TrackingOutputBuffers] = []
        for _ in 0..<maxFramesInFlight
        {
            guard let objectScoreLogit = commandQueue.device.makeBuffer(length: MemoryLayout<Float>.stride),
                  let objectPointers = commandQueue.device.makeBuffer(
                    length: Self.maskCount * Self.objectPointerWidth * MemoryLayout<Float>.stride
                  ) else
            {
                throw EfficientTAMError("Could not allocate EfficientTAM decoder tracking-output buffers.")
            }
            trackingOutputBuffers.append(
                TrackingOutputBuffers(objectScoreLogit: objectScoreLogit, objectPointers: objectPointers)
            )
        }
        self.trackingOutputBuffers = trackingOutputBuffers

        let weights = try EfficientTAMWeights(binaryURL: weightsBinaryURL, manifestURL: weightsManifestURL)
        let noMaskEmbedding = try weights.floats(named: "sam_prompt_encoder.no_mask_embed.weight")
        let spatialCount = EfficientTAMImageEncoder.embeddingWidth * EfficientTAMImageEncoder.embeddingHeight
        let tiledNoMaskEmbedding = noMaskEmbedding.flatMap
        {
            [Float](repeating: $0, count: spatialCount)
        }
        guard let noMaskEmbeddingBuffer = commandQueue.device.makeBuffer(
            bytes: tiledNoMaskEmbedding,
            length: tiledNoMaskEmbedding.count * MemoryLayout<Float>.stride
        ) else
        {
            throw EfficientTAMError("Could not allocate the EfficientTAM no-mask embedding buffer.")
        }
        self.noMaskEmbeddingBuffer = noMaskEmbeddingBuffer

        let builder = EfficientTAMPromptDecoderGraphBuilder(
            graph: self.graph,
            weights: weights,
            promptCount: promptCount
        )
        let imageEmbedding = self.graph.placeholder(
            shape: [1, 256, 32, 32],
            dataType: .float32,
            name: "image_embedding"
        )
        let promptCoordinates = self.graph.placeholder(
            shape: [1, promptCount as NSNumber, 2],
            dataType: .float32,
            name: "prompt_coordinates"
        )
        let promptLabels = self.graph.placeholder(
            shape: [1, promptCount as NSNumber],
            dataType: .float32,
            name: "prompt_labels"
        )
        let densePromptEmbedding = self.graph.placeholder(
            shape: [1, 256, 32, 32],
            dataType: .float32,
            name: "dense_prompt_embedding"
        )
        self.imageEmbeddingTensor = imageEmbedding
        self.promptCoordinatesTensor = promptCoordinates
        self.promptLabelsTensor = promptLabels
        self.densePromptEmbeddingTensor = densePromptEmbedding
        let outputs = try builder.build(
            imageEmbedding: imageEmbedding,
            promptCoordinates: promptCoordinates,
            promptLabels: promptLabels,
            densePromptEmbedding: densePromptEmbedding
        )
        self.maskLogitsTensor = outputs.maskLogits
        self.iouPredictionsTensor = outputs.iouPredictions
        self.objectScoreLogitTensor = outputs.objectScoreLogit
        self.objectPointersTensor = outputs.objectPointers

        let device = MPSGraphDevice(mtlDevice: commandQueue.device)
        let imageType = MPSGraphShapedType(shape: imageEmbedding.shape ?? [], dataType: .float32)
        let coordinatesType = MPSGraphShapedType(shape: promptCoordinates.shape ?? [], dataType: .float32)
        let labelsType = MPSGraphShapedType(shape: promptLabels.shape ?? [], dataType: .float32)
        let densePromptType = MPSGraphShapedType(shape: densePromptEmbedding.shape ?? [], dataType: .float32)
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level1
        descriptor.waitForCompilationCompletion = true
        self.executable = self.graph.compile(
            with: device,
            feeds: [
                imageEmbedding: imageType,
                promptCoordinates: coordinatesType,
                promptLabels: labelsType,
                densePromptEmbedding: densePromptType,
            ],
            targetTensors: [
                outputs.maskLogits,
                outputs.iouPredictions,
                outputs.objectScoreLogit,
                outputs.objectPointers,
            ],
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        self.executable.specialize(
            with: device,
            inputTypes: [imageType, coordinatesType, labelsType, densePromptType],
            compilationDescriptor: descriptor
        )
    }

    public func run(
        imageEmbeddingBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer,
        densePromptEmbeddingBuffer: MTLBuffer? = nil
    ) throws -> EfficientTAMMaskPrediction
    {
        try self.validate(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            promptCoordinatesBuffer: promptCoordinatesBuffer,
            promptLabelsBuffer: promptLabelsBuffer,
            densePromptEmbeddingBuffer: densePromptEmbeddingBuffer
        )
        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }
        let results = self.executable.run(
            with: self.commandQueue,
            inputs: self.inputs(
                imageEmbeddingBuffer: imageEmbeddingBuffer,
                promptCoordinatesBuffer: promptCoordinatesBuffer,
                promptLabelsBuffer: promptLabelsBuffer,
                densePromptEmbeddingBuffer: densePromptEmbeddingBuffer
            ),
            results: nil,
            executionDescriptor: nil
        )
        guard results.count == 4 else
        {
            throw EfficientTAMError("EfficientTAM prompt decoding produced \(results.count) outputs; expected 4.")
        }
        return self.prediction(from: results, slot: slot)
    }

    public func run(
        imageEmbeddingBuffer: MTLBuffer,
        prompts: [EfficientTAMPrompt],
        densePromptEmbeddingBuffer: MTLBuffer? = nil
    ) throws -> EfficientTAMMaskPrediction
    {
        let buffers = try self.makePromptBuffers(prompts)
        return try self.run(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            promptCoordinatesBuffer: buffers.coordinates,
            promptLabelsBuffer: buffers.labels,
            densePromptEmbeddingBuffer: densePromptEmbeddingBuffer
        )
    }

    @discardableResult
    public func submit(
        imageEmbeddingBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer,
        densePromptEmbeddingBuffer: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer,
        commit: Bool,
        completion: @escaping (Result<EfficientTAMMaskPrediction, any Error>) -> Void
    ) throws -> Bool
    {
        try self.validate(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            promptCoordinatesBuffer: promptCoordinatesBuffer,
            promptLabelsBuffer: promptLabelsBuffer,
            densePromptEmbeddingBuffer: densePromptEmbeddingBuffer,
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
            else if results.count == 4
            {
                completion(.success(self.prediction(from: results, slot: slot)))
            }
            else
            {
                completion(.failure(EfficientTAMError("EfficientTAM prompt decoding did not produce both outputs.")))
            }
        }
        let mpsCommandBuffer = EfficientTAMCommandBuffer.target(for: commandBuffer).commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: self.inputs(
                    imageEmbeddingBuffer: imageEmbeddingBuffer,
                    promptCoordinatesBuffer: promptCoordinatesBuffer,
                    promptLabelsBuffer: promptLabelsBuffer,
                    densePromptEmbeddingBuffer: densePromptEmbeddingBuffer
                ),
                results: nil,
                executionDescriptor: descriptor
            )
            if commit
            {
                mpsCommandBuffer.commit()
            }
        }
        return true
    }

    @discardableResult
    public func encode(
        imageEmbeddingBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer,
        densePromptEmbeddingBuffer: MTLBuffer? = nil,
        maskLogitsBuffer: MTLBuffer,
        iouPredictionsBuffer: MTLBuffer,
        objectScoreLogitBuffer: MTLBuffer? = nil,
        objectPointersBuffer: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> Bool
    {
        try self.validate(
            imageEmbeddingBuffer: imageEmbeddingBuffer,
            promptCoordinatesBuffer: promptCoordinatesBuffer,
            promptLabelsBuffer: promptLabelsBuffer,
            densePromptEmbeddingBuffer: densePromptEmbeddingBuffer,
            maskLogitsBuffer: maskLogitsBuffer,
            iouPredictionsBuffer: iouPredictionsBuffer,
            objectScoreLogitBuffer: objectScoreLogitBuffer,
            objectPointersBuffer: objectPointersBuffer,
            commandBuffer: commandBuffer
        )
        guard let slot = self.acquireSlotNonBlocking() else { return false }

        let maskData = MPSGraphTensorData(
            maskLogitsBuffer,
            shape: self.maskLogitsTensor.shape ?? [],
            dataType: .float32
        )
        let iouData = MPSGraphTensorData(
            iouPredictionsBuffer,
            shape: self.iouPredictionsTensor.shape ?? [],
            dataType: .float32
        )
        let trackingBuffers = self.trackingOutputBuffers[slot]
        let objectScoreData = MPSGraphTensorData(
            objectScoreLogitBuffer ?? trackingBuffers.objectScoreLogit,
            shape: self.objectScoreLogitTensor.shape ?? [],
            dataType: .float32
        )
        let objectPointersData = MPSGraphTensorData(
            objectPointersBuffer ?? trackingBuffers.objectPointers,
            shape: self.objectPointersTensor.shape ?? [],
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
                inputs: self.inputs(
                    imageEmbeddingBuffer: imageEmbeddingBuffer,
                    promptCoordinatesBuffer: promptCoordinatesBuffer,
                    promptLabelsBuffer: promptLabelsBuffer,
                    densePromptEmbeddingBuffer: densePromptEmbeddingBuffer
                ),
                results: [maskData, iouData, objectScoreData, objectPointersData],
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

    public func makePromptBuffers(
        _ prompts: [EfficientTAMPrompt]
    ) throws -> (coordinates: MTLBuffer, labels: MTLBuffer)
    {
        guard prompts.count == self.promptCount else
        {
            throw EfficientTAMError(
                "Received \(prompts.count) prompt tokens; this decoder was compiled for \(self.promptCount)."
            )
        }
        let coordinates = prompts.flatMap { [$0.x, $0.y] }
        let labels = prompts.map { $0.label.rawValue }
        guard let coordinatesBuffer = self.commandQueue.device.makeBuffer(
            bytes: coordinates,
            length: coordinates.count * MemoryLayout<Float>.stride
        ), let labelsBuffer = self.commandQueue.device.makeBuffer(
            bytes: labels,
            length: labels.count * MemoryLayout<Float>.stride
        ) else
        {
            throw EfficientTAMError("Could not allocate EfficientTAM prompt buffers.")
        }
        return (coordinatesBuffer, labelsBuffer)
    }

    private func inputs(
        imageEmbeddingBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer,
        densePromptEmbeddingBuffer: MTLBuffer?
    ) -> [MPSGraphTensorData]
    {
        [
            MPSGraphTensorData(imageEmbeddingBuffer, shape: self.imageEmbeddingTensor.shape ?? [], dataType: .float32),
            MPSGraphTensorData(promptCoordinatesBuffer, shape: self.promptCoordinatesTensor.shape ?? [], dataType: .float32),
            MPSGraphTensorData(promptLabelsBuffer, shape: self.promptLabelsTensor.shape ?? [], dataType: .float32),
            MPSGraphTensorData(
                densePromptEmbeddingBuffer ?? self.noMaskEmbeddingBuffer,
                shape: self.densePromptEmbeddingTensor.shape ?? [],
                dataType: .float32
            ),
        ]
    }

    private func validate(
        imageEmbeddingBuffer: MTLBuffer,
        promptCoordinatesBuffer: MTLBuffer,
        promptLabelsBuffer: MTLBuffer,
        densePromptEmbeddingBuffer: MTLBuffer? = nil,
        maskLogitsBuffer: MTLBuffer? = nil,
        iouPredictionsBuffer: MTLBuffer? = nil,
        objectScoreLogitBuffer: MTLBuffer? = nil,
        objectPointersBuffer: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws
    {
        guard imageEmbeddingBuffer.length >= self.imageEmbeddingBufferLength else
        {
            throw EfficientTAMError("The image-embedding buffer is too small.")
        }
        guard promptCoordinatesBuffer.length >= self.promptCoordinatesBufferLength else
        {
            throw EfficientTAMError("The prompt-coordinate buffer is too small.")
        }
        guard promptLabelsBuffer.length >= self.promptLabelsBufferLength else
        {
            throw EfficientTAMError("The prompt-label buffer is too small.")
        }
        if let densePromptEmbeddingBuffer,
           densePromptEmbeddingBuffer.length < self.densePromptEmbeddingBufferLength
        {
            throw EfficientTAMError("The dense-prompt embedding buffer is too small.")
        }
        if let maskLogitsBuffer, maskLogitsBuffer.length < self.maskLogitsBufferLength
        {
            throw EfficientTAMError("The mask-logits buffer is too small.")
        }
        if let iouPredictionsBuffer, iouPredictionsBuffer.length < self.iouPredictionsBufferLength
        {
            throw EfficientTAMError("The IoU-predictions buffer is too small.")
        }
        if let objectScoreLogitBuffer, objectScoreLogitBuffer.length < self.objectScoreLogitBufferLength
        {
            throw EfficientTAMError("The object-score logit buffer is too small.")
        }
        if let objectPointersBuffer, objectPointersBuffer.length < self.objectPointersBufferLength
        {
            throw EfficientTAMError("The object-pointers buffer is too small.")
        }
        if let commandBuffer, commandBuffer.device !== self.commandQueue.device
        {
            throw EfficientTAMError("The command buffer and EfficientTAM decoder use different Metal devices.")
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

    private func prediction(from results: [MPSGraphTensorData], slot: Int) -> EfficientTAMMaskPrediction
    {
        let cache = self.outputCaches[slot]
        let maskCount = Self.maskCount * Self.maskWidth * Self.maskHeight
        if cache.maskLogits.count != maskCount
        {
            cache.maskLogits = [Float](repeating: 0, count: maskCount)
        }
        if cache.iouPredictions.count != Self.maskCount
        {
            cache.iouPredictions = [Float](repeating: 0, count: Self.maskCount)
        }
        if cache.objectPointers.count != Self.maskCount * Self.objectPointerWidth
        {
            cache.objectPointers = [Float](repeating: 0, count: Self.maskCount * Self.objectPointerWidth)
        }
        cache.maskLogits.withUnsafeMutableBufferPointer { buffer in
            results[0].mpsndarray().readBytes(buffer.baseAddress!, strideBytes: nil)
        }
        cache.iouPredictions.withUnsafeMutableBufferPointer { buffer in
            results[1].mpsndarray().readBytes(buffer.baseAddress!, strideBytes: nil)
        }
        var objectScoreLogit: Float = 0
        withUnsafeMutableBytes(of: &objectScoreLogit)
        {
            results[2].mpsndarray().readBytes($0.baseAddress!, strideBytes: nil)
        }
        cache.objectPointers.withUnsafeMutableBufferPointer
        {
            results[3].mpsndarray().readBytes($0.baseAddress!, strideBytes: nil)
        }
        return EfficientTAMMaskPrediction(
            maskLogits: cache.maskLogits,
            iouPredictions: cache.iouPredictions,
            objectScoreLogit: objectScoreLogit,
            objectPointers: cache.objectPointers
        )
    }
}

private struct EfficientTAMPromptDecoderGraphBuilder
{
    let graph: MPSGraph
    let weights: EfficientTAMWeights
    let promptCount: Int

    func build(
        imageEmbedding: MPSGraphTensor,
        promptCoordinates: MPSGraphTensor,
        promptLabels: MPSGraphTensor,
        densePromptEmbedding: MPSGraphTensor
    ) throws -> (
        maskLogits: MPSGraphTensor,
        iouPredictions: MPSGraphTensor,
        objectScoreLogit: MPSGraphTensor,
        objectPointers: MPSGraphTensor
    )
    {
        let sparsePrompts = try self.promptEmbeddings(
            coordinates: promptCoordinates,
            labels: promptLabels
        )
        let outputTokens = try self.outputTokens()
        let queryPosition = self.graph.concatTensors([outputTokens, sparsePrompts], dimension: 1, name: nil)
        var queries = queryPosition

        var image = self.graph.addition(imageEmbedding, densePromptEmbedding, name: nil)
        image = self.graph.reshape(image, shape: [1, 256, 1024], name: nil)
        var keys = self.graph.transpose(image, permutation: [0, 2, 1], name: nil)
        let imagePosition = try self.densePositionEmbedding()

        for index in 0..<2
        {
            let prefix = "sam_mask_decoder.transformer.layers.\(index)"
            if index == 0
            {
                queries = try self.attention(
                    query: queries,
                    key: queries,
                    value: queries,
                    prefix: "\(prefix).self_attn",
                    numberOfHeads: 8
                )
            }
            else
            {
                let positionedQueries = self.graph.addition(queries, queryPosition, name: nil)
                let attention = try self.attention(
                    query: positionedQueries,
                    key: positionedQueries,
                    value: queries,
                    prefix: "\(prefix).self_attn",
                    numberOfHeads: 8
                )
                queries = self.graph.addition(queries, attention, name: nil)
            }
            queries = try self.layerNorm(queries, prefix: "\(prefix).norm1")

            let tokenToImage = try self.attention(
                query: self.graph.addition(queries, queryPosition, name: nil),
                key: self.graph.addition(keys, imagePosition, name: nil),
                value: keys,
                prefix: "\(prefix).cross_attn_token_to_image",
                numberOfHeads: 8
            )
            queries = try self.layerNorm(
                self.graph.addition(queries, tokenToImage, name: nil),
                prefix: "\(prefix).norm2"
            )

            var mlp = try self.linear(queries, prefix: "\(prefix).mlp.layers.0")
            mlp = self.graph.reLU(with: mlp, name: nil)
            mlp = try self.linear(mlp, prefix: "\(prefix).mlp.layers.1")
            queries = try self.layerNorm(
                self.graph.addition(queries, mlp, name: nil),
                prefix: "\(prefix).norm3"
            )

            let imageToToken = try self.attention(
                query: self.graph.addition(keys, imagePosition, name: nil),
                key: self.graph.addition(queries, queryPosition, name: nil),
                value: queries,
                prefix: "\(prefix).cross_attn_image_to_token",
                numberOfHeads: 8
            )
            keys = try self.layerNorm(
                self.graph.addition(keys, imageToToken, name: nil),
                prefix: "\(prefix).norm4"
            )
        }

        let finalAttention = try self.attention(
            query: self.graph.addition(queries, queryPosition, name: nil),
            key: self.graph.addition(keys, imagePosition, name: nil),
            value: keys,
            prefix: "sam_mask_decoder.transformer.final_attn_token_to_image",
            numberOfHeads: 8
        )
        queries = try self.layerNorm(
            self.graph.addition(queries, finalAttention, name: nil),
            prefix: "sam_mask_decoder.transformer.norm_final_attn"
        )

        let objectScoreToken = self.graph.reshape(
            self.graph.sliceTensor(queries, dimension: 1, start: 0, length: 1, name: nil),
            shape: [1, 256],
            name: nil
        )
        let iouToken = self.graph.reshape(
            self.graph.sliceTensor(queries, dimension: 1, start: 1, length: 1, name: nil),
            shape: [1, 256],
            name: nil
        )
        let maskTokens = self.graph.sliceTensor(queries, dimension: 1, start: 2, length: 4, name: nil)

        var upscaled = self.graph.transpose(keys, permutation: [0, 2, 1], name: nil)
        upscaled = self.graph.reshape(upscaled, shape: [1, 256, 32, 32], name: nil)
        upscaled = try self.transposeConvolution(
            upscaled,
            prefix: "sam_mask_decoder.output_upscaling.0",
            outputChannels: 64,
            outputSize: 64
        )
        upscaled = try self.layerNorm2D(upscaled, prefix: "sam_mask_decoder.output_upscaling.1")
        upscaled = self.gelu(upscaled)
        upscaled = try self.transposeConvolution(
            upscaled,
            prefix: "sam_mask_decoder.output_upscaling.3",
            outputChannels: 32,
            outputSize: 128
        )
        upscaled = self.gelu(upscaled)
        let flattenedUpscaled = self.graph.reshape(upscaled, shape: [1, 32, 16384], name: nil)

        var hypernetworks: [MPSGraphTensor] = []
        for index in 0..<4
        {
            var token = self.graph.sliceTensor(maskTokens, dimension: 1, start: index, length: 1, name: nil)
            token = self.graph.reshape(token, shape: [1, 256], name: nil)
            for layer in 0..<3
            {
                token = try self.linear(
                    token,
                    prefix: "sam_mask_decoder.output_hypernetworks_mlps.\(index).layers.\(layer)"
                )
                if layer < 2
                {
                    token = self.graph.reLU(with: token, name: nil)
                }
            }
            hypernetworks.append(self.graph.reshape(token, shape: [1, 1, 32], name: nil))
        }
        let hypernetwork = self.graph.concatTensors(hypernetworks, dimension: 1, name: nil)
        var masks = self.graph.matrixMultiplication(primary: hypernetwork, secondary: flattenedUpscaled, name: nil)
        masks = self.graph.reshape(masks, shape: [1, 4, 128, 128], name: nil)
        masks = self.graph.sliceTensor(masks, dimension: 1, start: 1, length: 3, name: "mask_logits")

        var iou = iouToken
        for layer in 0..<3
        {
            iou = try self.linear(iou, prefix: "sam_mask_decoder.iou_prediction_head.layers.\(layer)")
            if layer < 2
            {
                iou = self.graph.reLU(with: iou, name: nil)
            }
        }
        iou = self.graph.sigmoid(with: iou, name: nil)
        iou = self.graph.sliceTensor(iou, dimension: 1, start: 1, length: 3, name: "iou_predictions")

        var objectScore = objectScoreToken
        for layer in 0..<3
        {
            objectScore = try self.linear(
                objectScore,
                prefix: "sam_mask_decoder.pred_obj_score_head.layers.\(layer)"
            )
            if layer < 2
            {
                objectScore = self.graph.reLU(with: objectScore, name: nil)
            }
        }

        var objectPointers = self.graph.sliceTensor(maskTokens, dimension: 1, start: 1, length: 3, name: nil)
        for layer in 0..<3
        {
            objectPointers = try self.linear(objectPointers, prefix: "obj_ptr_proj.layers.\(layer)")
            if layer < 2
            {
                objectPointers = self.graph.reLU(with: objectPointers, name: nil)
            }
        }
        let objectPresent = self.graph.greaterThan(objectScore, self.scalar(0), name: nil)
        masks = self.graph.select(
            predicate: self.graph.reshape(objectPresent, shape: [1, 1, 1, 1], name: nil),
            trueTensor: masks,
            falseTensor: self.scalar(-1024),
            name: "visible_mask_logits"
        )
        let noObjectPointer = try self.weights.constant(self.graph, named: "no_obj_ptr")
        objectPointers = self.graph.select(
            predicate: objectPresent,
            trueTensor: objectPointers,
            falseTensor: noObjectPointer,
            name: "object_pointers"
        )
        return (masks, iou, objectScore, objectPointers)
    }

    private func promptEmbeddings(
        coordinates: MPSGraphTensor,
        labels: MPSGraphTensor
    ) throws -> MPSGraphTensor
    {
        var normalized = self.graph.addition(coordinates, self.scalar(0.5), name: nil)
        normalized = self.graph.division(normalized, self.scalar(512), name: nil)
        normalized = self.graph.subtraction(
            self.graph.multiplication(normalized, self.scalar(2), name: nil),
            self.scalar(1),
            name: nil
        )
        let gaussian = try self.weights.constant(
            self.graph,
            named: "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
        )
        var phases = self.graph.matrixMultiplication(primary: normalized, secondary: gaussian, name: nil)
        phases = self.graph.multiplication(phases, self.scalar(2 * .pi), name: nil)
        var embedding = self.graph.concatTensors(
            [self.graph.sin(with: phases, name: nil), self.graph.cos(with: phases, name: nil)],
            dimension: 2,
            name: nil
        )
        let predicates = self.graph.reshape(labels, shape: [1, self.promptCount as NSNumber, 1], name: nil)
        let paddingPredicate = self.graph.equal(predicates, self.scalar(-1), name: nil)
        let notAPoint = try self.weights.constant(self.graph, named: "sam_prompt_encoder.not_a_point_embed.weight")
        embedding = self.graph.select(
            predicate: paddingPredicate,
            trueTensor: notAPoint,
            falseTensor: embedding,
            name: nil
        )
        for label in 0..<4
        {
            let predicate = self.graph.equal(predicates, self.scalar(Float(label)), name: nil)
            let labelEmbedding = try self.weights.constant(
                self.graph,
                named: "sam_prompt_encoder.point_embeddings.\(label).weight"
            )
            embedding = self.graph.select(
                predicate: predicate,
                trueTensor: self.graph.addition(embedding, labelEmbedding, name: nil),
                falseTensor: embedding,
                name: nil
            )
        }
        return embedding
    }

    private func outputTokens() throws -> MPSGraphTensor
    {
        let object = try self.weights.constant(self.graph, named: "sam_mask_decoder.obj_score_token.weight")
        let iou = try self.weights.constant(self.graph, named: "sam_mask_decoder.iou_token.weight")
        let masks = try self.weights.constant(self.graph, named: "sam_mask_decoder.mask_tokens.weight")
        let combined = self.graph.concatTensors([object, iou, masks], dimension: 0, name: nil)
        return self.graph.reshape(combined, shape: [1, 6, 256], name: nil)
    }

    private func attention(
        query: MPSGraphTensor,
        key: MPSGraphTensor,
        value: MPSGraphTensor,
        prefix: String,
        numberOfHeads: Int
    ) throws -> MPSGraphTensor
    {
        var projectedQuery = try self.linear(query, prefix: "\(prefix).q_proj")
        var projectedKey = try self.linear(key, prefix: "\(prefix).k_proj")
        var projectedValue = try self.linear(value, prefix: "\(prefix).v_proj")
        guard let queryShape = projectedQuery.shape?.map(\.intValue),
              let keyShape = projectedKey.shape?.map(\.intValue),
              queryShape.count == 3, keyShape.count == 3 else
        {
            throw EfficientTAMError("Unexpected EfficientTAM decoder attention shape.")
        }
        let queryCount = queryShape[1]
        let keyCount = keyShape[1]
        let internalDimension = queryShape[2]
        let headDimension = internalDimension / numberOfHeads
        projectedQuery = self.graph.reshape(
            projectedQuery,
            shape: [1, queryCount as NSNumber, numberOfHeads as NSNumber, headDimension as NSNumber],
            name: nil
        )
        projectedKey = self.graph.reshape(
            projectedKey,
            shape: [1, keyCount as NSNumber, numberOfHeads as NSNumber, headDimension as NSNumber],
            name: nil
        )
        projectedValue = self.graph.reshape(
            projectedValue,
            shape: [1, keyCount as NSNumber, numberOfHeads as NSNumber, headDimension as NSNumber],
            name: nil
        )
        projectedQuery = self.graph.transpose(projectedQuery, permutation: [0, 2, 1, 3], name: nil)
        projectedKey = self.graph.transpose(projectedKey, permutation: [0, 2, 1, 3], name: nil)
        projectedValue = self.graph.transpose(projectedValue, permutation: [0, 2, 1, 3], name: nil)
        var output = EfficientTAMAttentionOps.attention(
            graph: self.graph,
            query: projectedQuery,
            key: projectedKey,
            value: projectedValue,
            mask: nil,
            scale: 1 / sqrt(Float(headDimension))
        )
        output = self.graph.transpose(output, permutation: [0, 2, 1, 3], name: nil)
        output = self.graph.reshape(output, shape: [1, queryCount as NSNumber, internalDimension as NSNumber], name: nil)
        return try self.linear(output, prefix: "\(prefix).out_proj")
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
            epsilon: 1e-5,
            name: nil
        )
    }

    private func layerNorm2D(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let mean = self.graph.mean(of: input, axes: [1], name: nil)
        let variance = self.graph.variance(of: input, mean: mean, axes: [1], name: nil)
        return self.graph.normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: try self.channelConstant(named: "\(prefix).weight"),
            beta: try self.channelConstant(named: "\(prefix).bias"),
            epsilon: 1e-6,
            name: nil
        )
    }

    private func transposeConvolution(
        _ input: MPSGraphTensor,
        prefix: String,
        outputChannels: Int,
        outputSize: Int
    ) throws -> MPSGraphTensor
    {
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 2,
            strideInY: 2,
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
            throw EfficientTAMError("Could not create EfficientTAM transpose-convolution descriptor.")
        }
        var output = self.graph.convolutionTranspose2D(
            input,
            weights: try self.weights.constant(self.graph, named: "\(prefix).weight"),
            outputShape: [1, outputChannels as NSNumber, outputSize as NSNumber, outputSize as NSNumber],
            descriptor: descriptor,
            name: nil
        )
        output = self.graph.addition(output, try self.channelConstant(named: "\(prefix).bias"), name: nil)
        return output
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

    private func densePositionEmbedding() throws -> MPSGraphTensor
    {
        let gaussian = try self.weights.floats(
            named: "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
        )
        var values = [Float](repeating: 0, count: 256 * 32 * 32)
        for y in 0..<32
        {
            for x in 0..<32
            {
                let normalizedX = 2 * ((Float(x) + 0.5) / 32) - 1
                let normalizedY = 2 * ((Float(y) + 0.5) / 32) - 1
                for feature in 0..<128
                {
                    let phase = 2 * Float.pi * (
                        normalizedX * gaussian[feature]
                            + normalizedY * gaussian[128 + feature]
                    )
                    values[(feature * 32 + y) * 32 + x] = sin(phase)
                    values[((feature + 128) * 32 + y) * 32 + x] = cos(phase)
                }
            }
        }
        let nchw = self.graph.constant(
            Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
            shape: [1, 256, 32, 32],
            dataType: .float32
        )
        let flat = self.graph.reshape(nchw, shape: [1, 256, 1024], name: nil)
        return self.graph.transpose(flat, permutation: [0, 2, 1], name: nil)
    }

    private func channelConstant(named name: String) throws -> MPSGraphTensor
    {
        let values = try self.weights.floats(named: name)
        return self.graph.constant(
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
