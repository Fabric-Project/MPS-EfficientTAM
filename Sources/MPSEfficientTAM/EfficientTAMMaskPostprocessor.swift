import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// EfficientTAM's official mask-logit postprocessing: bilinear resize with
/// half-pixel centers and unaligned corners. It intentionally leaves logits
/// unthresholded so downstream clients can compose their own refinement.
public final class EfficientTAMMaskPostprocessor
{
    public let maskCount: Int
    public let outputWidth: Int
    public let outputHeight: Int

    public var inputBufferLength: Int
    {
        self.maskCount
            * EfficientTAMPromptDecoder.maskWidth
            * EfficientTAMPromptDecoder.maskHeight
            * MemoryLayout<Float>.stride
    }

    public var outputBufferLength: Int
    {
        self.maskCount * self.outputWidth * self.outputHeight * MemoryLayout<Float>.stride
    }

    private let graph = MPSGraph()
    private let commandQueue: MTLCommandQueue
    private let inputTensor: MPSGraphTensor
    private let outputTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable
    private let slotSemaphore: DispatchSemaphore
    private let slotLock = NSLock()
    private var freeSlots: [Int]
    private let outputCaches: [OutputCache]

    private final class OutputCache
    {
        var values: [Float]

        init(count: Int)
        {
            self.values = [Float](repeating: 0, count: count)
        }
    }

    public init(
        maskCount: Int = EfficientTAMPromptDecoder.maskCount,
        outputWidth: Int,
        outputHeight: Int,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3
    ) throws
    {
        guard maskCount > 0, outputWidth > 0, outputHeight > 0 else
        {
            throw EfficientTAMError("EfficientTAM postprocessor dimensions and mask count must be positive.")
        }
        guard maxFramesInFlight > 0 else
        {
            throw EfficientTAMError("EfficientTAM maxFramesInFlight must be positive.")
        }
        self.maskCount = maskCount
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.commandQueue = commandQueue
        self.slotSemaphore = DispatchSemaphore(value: maxFramesInFlight)
        self.freeSlots = Array(0..<maxFramesInFlight)
        self.outputCaches = (0..<maxFramesInFlight).map
        {
            _ in OutputCache(count: maskCount * outputWidth * outputHeight)
        }

        let input = self.graph.placeholder(
            shape: [1, maskCount as NSNumber, 128, 128],
            dataType: .float32,
            name: "low_resolution_mask_logits"
        )
        self.inputTensor = input
        self.outputTensor = self.graph.resize(
            input,
            size: [outputHeight as NSNumber, outputWidth as NSNumber],
            mode: .bilinear,
            centerResult: true,
            alignCorners: false,
            layout: .NCHW,
            name: "resized_mask_logits"
        )

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
            throw EfficientTAMError("EfficientTAM mask postprocessing produced no output tensor.")
        }
        return self.read(result, slot: slot)
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
                completion(.success(self.read(result, slot: slot)))
            }
            else
            {
                completion(.failure(EfficientTAMError("EfficientTAM mask postprocessing produced no output tensor.")))
            }
        }
        let input = MPSGraphTensorData(maskLogitsBuffer, shape: self.inputTensor.shape ?? [], dataType: .float32)
        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
        _ = self.executable.encode(to: mpsCommandBuffer, inputs: [input], results: nil, executionDescriptor: descriptor)
        if commit { mpsCommandBuffer.commit() }
        return true
    }

    @discardableResult
    public func encode(
        maskLogitsBuffer: MTLBuffer,
        resizedMaskLogitsBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> Bool
    {
        try self.validate(input: maskLogitsBuffer, output: resizedMaskLogitsBuffer, commandBuffer: commandBuffer)
        guard let slot = self.acquireSlotNonBlocking() else { return false }
        commandBuffer.addCompletedHandler { [weak self] _ in self?.releaseSlot(slot) }
        let input = MPSGraphTensorData(maskLogitsBuffer, shape: self.inputTensor.shape ?? [], dataType: .float32)
        let output = MPSGraphTensorData(
            resizedMaskLogitsBuffer,
            shape: self.outputTensor.shape ?? [],
            dataType: .float32
        )
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
        _ = self.executable.encode(to: mpsCommandBuffer, inputs: [input], results: [output], executionDescriptor: descriptor)
        if commit { mpsCommandBuffer.commit() }
        return true
    }

    /// Converts resized logits to a compact binary mask after readback.
    public static func binaryMask(from logits: [Float], threshold: Float = 0) -> [UInt8]
    {
        logits.map { $0 > threshold ? 255 : 0 }
    }

    private func validate(
        input: MTLBuffer,
        output: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws
    {
        guard input.length >= self.inputBufferLength else
        {
            throw EfficientTAMError("The low-resolution mask-logits buffer is too small.")
        }
        if let output, output.length < self.outputBufferLength
        {
            throw EfficientTAMError("The resized mask-logits buffer is too small.")
        }
        if let commandBuffer, commandBuffer.device !== self.commandQueue.device
        {
            throw EfficientTAMError("The command buffer and EfficientTAM postprocessor use different Metal devices.")
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
        self.slotSemaphore.wait()
        self.slotLock.lock()
        defer { self.slotLock.unlock() }
        return self.freeSlots.removeLast()
    }

    private func acquireSlotNonBlocking() -> Int?
    {
        guard self.slotSemaphore.wait(timeout: .now()) == .success else { return nil }
        self.slotLock.lock()
        defer { self.slotLock.unlock() }
        return self.freeSlots.removeLast()
    }

    private func releaseSlot(_ slot: Int)
    {
        self.slotLock.lock()
        self.freeSlots.append(slot)
        self.slotLock.unlock()
        self.slotSemaphore.signal()
    }
}
