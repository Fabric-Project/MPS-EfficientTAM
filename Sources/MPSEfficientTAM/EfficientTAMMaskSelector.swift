import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public struct EfficientTAMSelectedMask: Sendable
{
    public let maskLogits: [Float]
    public let iouPrediction: Float
    public let objectPointer: [Float]
}

/// Selects the highest-IoU mask and its aligned object pointer entirely on GPU.
public final class EfficientTAMMaskSelector
{
    public var maskLogitsBufferLength: Int { 3 * 128 * 128 * MemoryLayout<Float>.stride }
    public var iouPredictionsBufferLength: Int { 3 * MemoryLayout<Float>.stride }
    public var objectPointersBufferLength: Int { 3 * 256 * MemoryLayout<Float>.stride }
    public var selectedMaskLogitsBufferLength: Int { 128 * 128 * MemoryLayout<Float>.stride }
    public var selectedIoUPredictionBufferLength: Int { MemoryLayout<Float>.stride }
    public var selectedObjectPointerBufferLength: Int { 256 * MemoryLayout<Float>.stride }

    private let graph = MPSGraph()
    private let commandQueue: MTLCommandQueue
    private let inputTensors: [MPSGraphTensor]
    private let outputTensors: [MPSGraphTensor]
    private let executable: MPSGraphExecutable
    private let slotPool: EfficientTAMSlotPool
    private let caches: [OutputCache]

    private final class OutputCache
    {
        var mask = [Float](repeating: 0, count: 128 * 128)
        var pointer = [Float](repeating: 0, count: 256)
    }

    public init(commandQueue: MTLCommandQueue, maxFramesInFlight: Int = 3) throws
    {
        guard maxFramesInFlight > 0 else
        {
            throw EfficientTAMError("EfficientTAM maxFramesInFlight must be positive.")
        }
        self.commandQueue = commandQueue
        self.slotPool = EfficientTAMSlotPool(count: maxFramesInFlight)
        self.caches = (0..<maxFramesInFlight).map { _ in OutputCache() }

        let masks = self.graph.placeholder(shape: [1, 3, 128, 128], dataType: .float32, name: "mask_logits")
        let iou = self.graph.placeholder(shape: [1, 3], dataType: .float32, name: "iou_predictions")
        let pointers = self.graph.placeholder(shape: [1, 3, 256], dataType: .float32, name: "object_pointers")
        self.inputTensors = [masks, iou, pointers]

        var selectedMask = self.graph.sliceTensor(masks, dimension: 1, start: 0, length: 1, name: nil)
        var selectedIoU = self.graph.sliceTensor(iou, dimension: 1, start: 0, length: 1, name: nil)
        var selectedPointer = self.graph.sliceTensor(pointers, dimension: 1, start: 0, length: 1, name: nil)
        for index in 1..<3
        {
            let candidateIoU = self.graph.sliceTensor(iou, dimension: 1, start: index, length: 1, name: nil)
            let useCandidate = self.graph.greaterThan(candidateIoU, selectedIoU, name: nil)
            let maskPredicate = self.graph.reshape(useCandidate, shape: [1, 1, 1, 1], name: nil)
            let pointerPredicate = self.graph.reshape(useCandidate, shape: [1, 1, 1], name: nil)
            selectedMask = self.graph.select(
                predicate: maskPredicate,
                trueTensor: self.graph.sliceTensor(masks, dimension: 1, start: index, length: 1, name: nil),
                falseTensor: selectedMask,
                name: nil
            )
            selectedPointer = self.graph.select(
                predicate: pointerPredicate,
                trueTensor: self.graph.sliceTensor(pointers, dimension: 1, start: index, length: 1, name: nil),
                falseTensor: selectedPointer,
                name: nil
            )
            selectedIoU = self.graph.select(
                predicate: useCandidate,
                trueTensor: candidateIoU,
                falseTensor: selectedIoU,
                name: nil
            )
        }
        selectedPointer = self.graph.reshape(selectedPointer, shape: [1, 256], name: "selected_object_pointer")
        self.outputTensors = [selectedMask, selectedIoU, selectedPointer]

        let device = MPSGraphDevice(mtlDevice: commandQueue.device)
        let inputTypes = self.inputTensors.map { MPSGraphShapedType(shape: $0.shape ?? [], dataType: .float32) }
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level1
        descriptor.waitForCompilationCompletion = true
        self.executable = self.graph.compile(
            with: device,
            feeds: Dictionary(uniqueKeysWithValues: zip(self.inputTensors, inputTypes)),
            targetTensors: self.outputTensors,
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        self.executable.specialize(with: device, inputTypes: inputTypes, compilationDescriptor: descriptor)
    }

    public func run(
        maskLogitsBuffer: MTLBuffer,
        iouPredictionsBuffer: MTLBuffer,
        objectPointersBuffer: MTLBuffer
    ) throws -> EfficientTAMSelectedMask
    {
        try self.validate(masks: maskLogitsBuffer, iou: iouPredictionsBuffer, pointers: objectPointersBuffer)
        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }
        let results = self.executable.run(
            with: self.commandQueue,
            inputs: self.inputs(masks: maskLogitsBuffer, iou: iouPredictionsBuffer, pointers: objectPointersBuffer),
            results: nil,
            executionDescriptor: nil
        )
        guard results.count == 3 else
        {
            throw EfficientTAMError("EfficientTAM mask selection did not produce all outputs.")
        }
        return self.selection(from: results, slot: slot)
    }

    @discardableResult
    public func submit(
        maskLogitsBuffer: MTLBuffer,
        iouPredictionsBuffer: MTLBuffer,
        objectPointersBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool,
        completion: @escaping (Result<EfficientTAMSelectedMask, any Error>) -> Void
    ) throws -> Bool
    {
        try self.validate(
            masks: maskLogitsBuffer,
            iou: iouPredictionsBuffer,
            pointers: objectPointersBuffer,
            commandBuffer: commandBuffer
        )
        guard let slot = self.acquireSlotNonBlocking() else { return false }
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        descriptor.completionHandler = { [weak self] results, error in
            guard let self else { return }
            defer { self.releaseSlot(slot) }
            if let error { completion(.failure(error)) }
            else if results.count == 3 { completion(.success(self.selection(from: results, slot: slot))) }
            else { completion(.failure(EfficientTAMError("EfficientTAM mask selection did not produce all outputs."))) }
        }
        let mpsCommandBuffer = EfficientTAMCommandBuffer.target(for: commandBuffer).commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: self.inputs(masks: maskLogitsBuffer, iou: iouPredictionsBuffer, pointers: objectPointersBuffer),
                results: nil,
                executionDescriptor: descriptor
            )
            if commit { mpsCommandBuffer.commit() }
        }
        return true
    }

    @discardableResult
    public func encode(
        maskLogitsBuffer: MTLBuffer,
        iouPredictionsBuffer: MTLBuffer,
        objectPointersBuffer: MTLBuffer,
        selectedMaskLogitsBuffer: MTLBuffer,
        selectedIoUPredictionBuffer: MTLBuffer,
        selectedObjectPointerBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
        commit: Bool
    ) throws -> Bool
    {
        try self.validate(
            masks: maskLogitsBuffer,
            iou: iouPredictionsBuffer,
            pointers: objectPointersBuffer,
            selectedMask: selectedMaskLogitsBuffer,
            selectedIoU: selectedIoUPredictionBuffer,
            selectedPointer: selectedObjectPointerBuffer,
            commandBuffer: commandBuffer
        )
        guard let slot = self.acquireSlotNonBlocking() else { return false }
        let outputs = zip(
            [selectedMaskLogitsBuffer, selectedIoUPredictionBuffer, selectedObjectPointerBuffer],
            self.outputTensors
        ).map { MPSGraphTensorData($0.0, shape: $0.1.shape ?? [], dataType: .float32) }
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        let target = EfficientTAMCommandBuffer.target(for: commandBuffer)
        let mpsCommandBuffer = target.commandBuffer
        autoreleasepool
        {
            _ = self.executable.encode(
                to: mpsCommandBuffer,
                inputs: self.inputs(masks: maskLogitsBuffer, iou: iouPredictionsBuffer, pointers: objectPointersBuffer),
                results: outputs,
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

    private func inputs(masks: MTLBuffer, iou: MTLBuffer, pointers: MTLBuffer) -> [MPSGraphTensorData]
    {
        zip([masks, iou, pointers], self.inputTensors).map
        {
            MPSGraphTensorData($0.0, shape: $0.1.shape ?? [], dataType: .float32)
        }
    }

    private func validate(
        masks: MTLBuffer,
        iou: MTLBuffer,
        pointers: MTLBuffer,
        selectedMask: MTLBuffer? = nil,
        selectedIoU: MTLBuffer? = nil,
        selectedPointer: MTLBuffer? = nil,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws
    {
        guard masks.length >= self.maskLogitsBufferLength,
              iou.length >= self.iouPredictionsBufferLength,
              pointers.length >= self.objectPointersBufferLength else
        {
            throw EfficientTAMError("An EfficientTAM mask-selector input buffer is too small.")
        }
        if let selectedMask, selectedMask.length < self.selectedMaskLogitsBufferLength
        {
            throw EfficientTAMError("The selected mask-logits buffer is too small.")
        }
        if let selectedIoU, selectedIoU.length < self.selectedIoUPredictionBufferLength
        {
            throw EfficientTAMError("The selected IoU buffer is too small.")
        }
        if let selectedPointer, selectedPointer.length < self.selectedObjectPointerBufferLength
        {
            throw EfficientTAMError("The selected object-pointer buffer is too small.")
        }
        if let commandBuffer, commandBuffer.device !== self.commandQueue.device
        {
            throw EfficientTAMError("The command buffer and EfficientTAM mask selector use different Metal devices.")
        }
    }

    private func selection(from results: [MPSGraphTensorData], slot: Int) -> EfficientTAMSelectedMask
    {
        let cache = self.caches[slot]
        cache.mask.withUnsafeMutableBufferPointer { results[0].mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        var iou: Float = 0
        withUnsafeMutableBytes(of: &iou) { results[1].mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        cache.pointer.withUnsafeMutableBufferPointer { results[2].mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return EfficientTAMSelectedMask(maskLogits: cache.mask, iouPrediction: iou, objectPointer: cache.pointer)
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
