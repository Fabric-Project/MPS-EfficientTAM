import Foundation
import Metal
import MetalPerformanceShaders
import Testing
@testable import MPSEfficientTAM

private let frameCount = 11
/// The fixture blacks out this frame so the object-absent path runs.
private let occludedFrameIndex = 9
private let frameWidth = 512
private let frameHeight = 512

/// Runs the tracker over eleven frames with no CPU/GPU waits between
/// submissions, then compares every frame's selected mask, IoU, object score,
/// object pointer and new spatial memory against upstream `track_step` output.
/// Frame N attends to min(N, 7) spatial memories and N object pointers, so this
/// exercises the (1,1), (2,2) ... (7,7) and (7,8)...(7,10) attention shapes and
/// the temporal-position ordering. Frame 9 is black (object absent), and frame
/// 10 must recover from the absent-object memory and pointer.
@Test func videoTrackerMatchesOfficialPyTorchOverElevenFrames() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let tracker = try EfficientTAMVideoTracker(
        commandQueue: commandQueue,
        maxFramesInFlight: frameCount
    )
    let frames = try trackerFixtureBytes(named: "tracker_frames_rgb_uint8")
    let frameLength = frameWidth * frameHeight * 3
    #expect(frames.count == frameCount * frameLength)

    let inputBuffers: [MTLBuffer] = try (0..<frameCount).map
    {
        index in
        let rgb = frames[(index * frameLength)..<((index + 1) * frameLength)].map { Float($0) / 255 }
        return try #require(
            device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride)
        )
    }

    var outputs: [EfficientTAMVideoTrackingOutput] = []
    let initial = try tracker.encodeInitialFrame(
        inputBuffer: inputBuffers[0],
        prompts: [
            .init(x: 160, y: 300, label: .positivePoint),
            .init(x: 0, y: 0, label: .padding),
        ]
    )
    outputs.append(try #require(initial))
    for index in 1..<frameCount
    {
        outputs.append(try #require(try tracker.encodeNextFrame(inputBuffer: inputBuffers[index])))
    }
    #expect(outputs.map(\.frameIndex) == Array(0..<frameCount))

    try verifyTrackerOutputs(outputs, device: device, commandQueue: commandQueue)
}

/// The Fabric-style flow: one `MPSCommandBuffer` per frame, "upstream" GPU work
/// that produces the model input encoded onto it first, the tracker after that
/// with `commit: false`, and a single commit by the owner. Nothing here waits
/// between frames. MPSGraph splits these graphs across command buffers, so this
/// also exercises the split-safe path.
@Test func videoTrackerEncodesOntoACallerOwnedMPSCommandBuffer() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let tracker = try EfficientTAMVideoTracker(commandQueue: commandQueue, maxFramesInFlight: frameCount)
    let frames = try trackerFixtureBytes(named: "tracker_frames_rgb_uint8")
    let frameLength = frameWidth * frameHeight * 3
    let inputBytes = frameLength * MemoryLayout<Float>.stride

    var outputs: [EfficientTAMVideoTrackingOutput] = []
    for index in 0..<frameCount
    {
        let rgb = frames[(index * frameLength)..<((index + 1) * frameLength)].map { Float($0) / 255 }
        let staging = try #require(device.makeBuffer(bytes: rgb, length: inputBytes, options: .storageModeShared))
        let modelInput = try #require(device.makeBuffer(length: inputBytes, options: .storageModePrivate))

        let rawCommandBuffer = try #require(commandQueue.makeCommandBuffer())
        let frameCommandBuffer = MPSCommandBuffer(commandBuffer: rawCommandBuffer)

        // Upstream work on the shared buffer: the tracker's input only exists
        // once this blit has run, so the tracker must be ordered after it.
        let upstream = try #require(frameCommandBuffer.makeBlitCommandEncoder())
        upstream.copy(from: staging, sourceOffset: 0, to: modelInput, destinationOffset: 0, size: inputBytes)
        upstream.endEncoding()

        let output: EfficientTAMVideoTrackingOutput?
        if index == 0
        {
            output = try tracker.encodeInitialFrame(
                inputBuffer: modelInput,
                prompts: [
                    .init(x: 160, y: 300, label: .positivePoint),
                    .init(x: 0, y: 0, label: .padding),
                ],
                commandBuffer: frameCommandBuffer,
                commit: false
            )
        }
        else
        {
            output = try tracker.encodeNextFrame(inputBuffer: modelInput, commandBuffer: frameCommandBuffer, commit: false)
        }
        outputs.append(try #require(output))

        // The owner commits, exactly once, through the wrapper that survived any splits.
        frameCommandBuffer.commit()
    }
    try verifyTrackerOutputs(outputs, device: device, commandQueue: commandQueue)
}

/// Two trackers on one device share their stateless stages (encoder, decoder,
/// selector, memory encoder, attention) and each keeps its own memory bank.
/// Interleaving their frames on the same shared stages must leave BOTH matching
/// the official reference, frame for frame. That is the re-entrancy guarantee:
/// a shared stage keeps no per-tracker state, and its in-flight capacity covers
/// both trackers at once.
@Test func twoTrackersSharingStagesBothMatchOfficialPyTorch() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let first = try EfficientTAMVideoTracker(commandQueue: commandQueue, maxFramesInFlight: 3)
    let second = try EfficientTAMVideoTracker(commandQueue: commandQueue, maxFramesInFlight: 3)
    #expect(first.sharedStagesIdentifier == second.sharedStagesIdentifier, "trackers on one device must share their stages")

    let frames = try trackerFixtureBytes(named: "tracker_frames_rgb_uint8")
    let frameLength = frameWidth * frameHeight * 3
    let inputBuffers: [MTLBuffer] = try (0..<frameCount).map
    {
        index in
        let rgb = frames[(index * frameLength)..<((index + 1) * frameLength)].map { Float($0) / 255 }
        return try #require(device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride))
    }
    let prompts: [EfficientTAMPrompt] = [
        .init(x: 160, y: 300, label: .positivePoint),
        .init(x: 0, y: 0, label: .padding),
    ]

    // A tracker returns nil while its own frames are in flight, and its slot
    // returns from a completion handler, so retry briefly.
    func submit(_ tracker: EfficientTAMVideoTracker, frame: Int) throws -> EfficientTAMVideoTrackingOutput
    {
        for _ in 0..<2000
        {
            let output = frame == 0
                ? try tracker.encodeInitialFrame(inputBuffer: inputBuffers[0], prompts: prompts)
                : try tracker.encodeNextFrame(inputBuffer: inputBuffers[frame])
            if let output { return output }
            Thread.sleep(forTimeInterval: 0.001)
        }
        throw EfficientTAMError("Tracker never accepted frame \(frame).")
    }

    var firstOutputs: [EfficientTAMVideoTrackingOutput] = []
    var secondOutputs: [EfficientTAMVideoTrackingOutput] = []
    for index in 0..<frameCount
    {
        firstOutputs.append(try submit(first, frame: index))
        secondOutputs.append(try submit(second, frame: index))
    }
    try verifyTrackerOutputs(firstOutputs, device: device, commandQueue: commandQueue)
    try verifyTrackerOutputs(secondOutputs, device: device, commandQueue: commandQueue)
}

/// Reads every output back with one blit pass and one wait (test-only), and
/// compares each frame to the upstream `track_step` reference.
private func verifyTrackerOutputs(
    _ outputs: [EfficientTAMVideoTrackingOutput],
    device: MTLDevice,
    commandQueue: MTLCommandQueue
) throws
{
    // One blit pass and one wait for the whole sequence.
    let maskLength = 128 * 128
    let pointerLength = 256
    let memoryLength = 64 * 32 * 32
    let perFrameFloats = maskLength + 1 + 1 + pointerLength + memoryLength
    let stride = MemoryLayout<Float>.stride
    let staging = try #require(
        device.makeBuffer(length: frameCount * perFrameFloats * stride, options: .storageModeShared)
    )
    let readback = try #require(commandQueue.makeCommandBuffer())
    let blit = try #require(readback.makeBlitCommandEncoder())
    for (index, output) in outputs.enumerated()
    {
        var offset = index * perFrameFloats * stride
        for (buffer, floats) in [
            (output.maskLogitsBuffer, maskLength),
            (output.iouPredictionBuffer, 1),
            (output.objectScoreLogitBuffer, 1),
            (output.objectPointerBuffer, pointerLength),
            (output.memoryFeaturesBuffer, memoryLength),
        ]
        {
            blit.copy(from: buffer, sourceOffset: 0, to: staging, destinationOffset: offset, size: floats * stride)
            offset += floats * stride
        }
    }
    blit.endEncoding()
    readback.commit()
    readback.waitUntilCompleted()
    #expect(readback.status == .completed)

    let values = UnsafeBufferPointer(
        start: staging.contents().assumingMemoryBound(to: Float.self),
        count: frameCount * perFrameFloats
    )
    let referenceMasks = try trackerFixtureFloats(named: "tracker_masks_reference")
    let referenceIoU = try trackerFixtureFloats(named: "tracker_iou_reference")
    let referenceScores = try trackerFixtureFloats(named: "tracker_object_score_reference")
    let referencePointers = try trackerFixtureFloats(named: "tracker_object_pointer_reference")
    let referenceMemories = try trackerFixtureFloats(named: "tracker_memory_features_reference")
    #expect(referenceMasks.count == frameCount * maskLength)
    #expect(referenceMemories.count == frameCount * memoryLength)
    #expect(referenceScores[occludedFrameIndex] < 0, "fixture must contain an object-absent frame")
    #expect(referenceScores.enumerated().allSatisfy { $0.offset == occludedFrameIndex || $0.element > 0 })

    for index in 0..<frameCount
    {
        let base = index * perFrameFloats
        let mask = values[base..<(base + maskLength)]
        let iou = values[base + maskLength]
        let score = values[base + maskLength + 1]
        let pointer = values[(base + maskLength + 2)..<(base + maskLength + 2 + pointerLength)]
        let memory = values[(base + maskLength + 2 + pointerLength)..<(base + perFrameFloats)]

        let maskError = errors(mask, referenceMasks[(index * maskLength)..<((index + 1) * maskLength)])
        let pointerError = errors(pointer, referencePointers[(index * pointerLength)..<((index + 1) * pointerLength)])
        let memoryError = errors(memory, referenceMemories[(index * memoryLength)..<((index + 1) * memoryLength)])
        let iouError = abs(iou - referenceIoU[index])
        let scoreError = abs(score - referenceScores[index])
        print(
            "Tracker frame \(index): mask MAE=\(maskError.mean) max=\(maskError.max), "
                + "IoU err=\(iouError), score err=\(scoreError), "
                + "pointer MAE=\(pointerError.mean) max=\(pointerError.max), "
                + "memory MAE=\(memoryError.mean) max=\(memoryError.max)"
        )
        // Observed errors are ~1e-5 (mask/memory) and ~1e-6 (pointer/score);
        // bounds leave roughly 5-10x headroom.
        #expect(maskError.mean < 5e-5, "frame \(index) mask MAE")
        #expect(maskError.max < 1e-3, "frame \(index) mask max error")
        #expect(iouError < 1e-4, "frame \(index) IoU")
        #expect(scoreError < 1e-3, "frame \(index) object score")
        #expect(pointerError.max < 1e-4, "frame \(index) object pointer")
        #expect(memoryError.mean < 1e-5, "frame \(index) memory MAE")
        #expect(memoryError.max < 3e-4, "frame \(index) memory max error")
        if index == occludedFrameIndex
        {
            #expect(score < 0, "object should be absent on the occluded frame")
            #expect(mask.allSatisfy { $0 == -1024 }, "absent object must suppress the whole mask")
        }
        else
        {
            #expect(score > 0, "frame \(index) object should be present")
        }
    }
}

/// Submits faster than the GPU drains. Saturation must drop frames (nil) rather
/// than throw or stall, dropped frames must not consume frame indexes, results
/// must stay finite, and once the GPU drains the tracker must accept work
/// again, which proves no in-flight slot leaked.
@Test(arguments: [1, 2, 3])
func videoTrackerBackpressureDropsWithoutThrowingOrLeakingSlots(maxFramesInFlight: Int) throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let tracker = try EfficientTAMVideoTracker(commandQueue: commandQueue, maxFramesInFlight: maxFramesInFlight)
    let prewarmStart = Date()
    try tracker.prewarmMemoryAttention()
    print("Prewarm (masked attention): \(Date().timeIntervalSince(prewarmStart)) s")

    let frames = try trackerFixtureBytes(named: "tracker_frames_rgb_uint8")
    let frameLength = frameWidth * frameHeight * 3
    let inputBuffers: [MTLBuffer] = try (0..<4).map
    {
        index in
        let rgb = frames[(index * frameLength)..<((index + 1) * frameLength)].map { Float($0) / 255 }
        return try #require(device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride))
    }

    let prompts: [EfficientTAMPrompt] = [
        .init(x: 160, y: 300, label: .positivePoint),
        .init(x: 0, y: 0, label: .padding),
    ]
    var last = try #require(try tracker.encodeInitialFrame(inputBuffer: inputBuffers[0], prompts: prompts))
    var accepted = 1
    var dropped = 0
    // Spin without waiting until 40 frames are accepted (past frame 16, so the
    // steady-state (7, 16) attention shape runs) or a time limit passes.
    let deadline = Date().addingTimeInterval(10)
    var attempt = 0
    while accepted < 40, Date() < deadline
    {
        if let output = try tracker.encodeNextFrame(inputBuffer: inputBuffers[attempt % inputBuffers.count])
        {
            last = output
            accepted += 1
        }
        else
        {
            dropped += 1
        }
        attempt += 1
    }
    #expect(accepted == 40, "tracker stalled instead of draining")
    print("Backpressure (maxFramesInFlight=\(maxFramesInFlight)): accepted=\(accepted) dropped=\(dropped)")
    #expect(last.frameIndex == accepted - 1, "dropped frames must not consume frame indexes")

    let staging = try #require(device.makeBuffer(length: 128 * 128 * 4 + 4, options: .storageModeShared))
    let readback = try #require(commandQueue.makeCommandBuffer())
    let blit = try #require(readback.makeBlitCommandEncoder())
    blit.copy(from: last.maskLogitsBuffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: 128 * 128 * 4)
    blit.copy(from: last.objectScoreLogitBuffer, sourceOffset: 0, to: staging, destinationOffset: 128 * 128 * 4, size: 4)
    blit.endEncoding()
    readback.commit()
    readback.waitUntilCompleted()
    #expect(readback.status == .completed)
    let values = UnsafeBufferPointer(
        start: staging.contents().assumingMemoryBound(to: Float.self),
        count: 128 * 128 + 1
    )
    #expect(values.allSatisfy { $0.isFinite })

    // Completion handlers run on a Metal callback thread shortly after the
    // GPU finishes; poll briefly for the slots to come back.
    var reaccepted = false
    for _ in 0..<200
    {
        if try tracker.encodeNextFrame(inputBuffer: inputBuffers[0]) != nil
        {
            reaccepted = true
            break
        }
        Thread.sleep(forTimeInterval: 0.01)
    }
    #expect(reaccepted, "tracker must accept work again after the GPU drains")
    // The re-accepted frame is deliberately still in flight as `tracker` goes
    // out of scope: tearing down mid-flight must not trap.
}

private func errors(
    _ actual: some Collection<Float>,
    _ reference: some Collection<Float>
) -> (mean: Float, max: Float)
{
    var sum: Float = 0
    var maximum: Float = 0
    var count = 0
    for (a, r) in zip(actual, reference)
    {
        let error = abs(a - r)
        sum += error
        maximum = Swift.max(maximum, error)
        count += 1
    }
    return (count == 0 ? .infinity : sum / Float(count), count == 0 ? .infinity : maximum)
}

private func trackerFixtureBytes(named name: String) throws -> [UInt8]
{
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: "bin", subdirectory: "Fixtures")
    )
    return [UInt8](try Data(contentsOf: url))
}

private func trackerFixtureFloats(named name: String) throws -> [Float]
{
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: "bin", subdirectory: "Fixtures")
    )
    return try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
