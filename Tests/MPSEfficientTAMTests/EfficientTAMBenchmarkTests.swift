import Foundation
import Metal
import QuartzCore
import Testing
@testable import MPSEfficientTAM

private let benchmarkEnabled = ProcessInfo.processInfo.environment["EFFICIENTTAM_BENCHMARK"] != nil

/// Opt-in: `EFFICIENTTAM_BENCHMARK=1 swift test -c release --filter trackerBenchmark`.
/// Prints numbers only and asserts nothing about speed.
@Test(.enabled(if: benchmarkEnabled))
func trackerBenchmark() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let inputBuffers = try benchmarkInputBuffers(device: device)
    let prompts: [EfficientTAMPrompt] = [
        .init(x: 160, y: 300, label: .positivePoint),
        .init(x: 0, y: 0, label: .padding),
    ]
    print("Benchmark device: \(device.name); footprint before any tracker: \(Int(physicalFootprintMB())) MB")

    // (frames in flight, command-queue command-buffer cap; 0 = the default queue)
    for (maxFramesInFlight, commandBufferCap) in [(1, 0), (2, 0), (3, 0), (3, 1024)]
    {
        let commandQueue = commandBufferCap == 0
            ? commandQueue
            : device.makeCommandQueue(maxCommandBufferCount: commandBufferCap) ?? commandQueue
        print("--- maxFramesInFlight=\(maxFramesInFlight), command buffer cap \(commandBufferCap == 0 ? "default" : String(commandBufferCap))")
        let cpu = StageTimings()
        do
        {
            let tracker = try EfficientTAMVideoTracker(commandQueue: commandQueue, maxFramesInFlight: maxFramesInFlight)
            tracker.cpuTimingHandler = { _, stage, milliseconds in cpu.add(stage, milliseconds) }
            try tracker.prewarmMemoryAttention()

            let frameTotal = 120
            var submitMilliseconds: [Double] = []
            let start = CACurrentMediaTime()
            var last = try #require(try tracker.encodeInitialFrame(inputBuffer: inputBuffers[0], prompts: prompts))
            var accepted = 1
            while accepted < frameTotal
            {
                let submitStart = CACurrentMediaTime()
                let output = try tracker.encodeNextFrame(inputBuffer: inputBuffers[accepted % inputBuffers.count])
                if let output
                {
                    submitMilliseconds.append((CACurrentMediaTime() - submitStart) * 1000)
                    last = output
                    accepted += 1
                }
            }
            // Benchmark-only: one wait for the whole sequence, an empty command
            // buffer queued behind everything the tracker submitted.
            let drain = try #require(commandQueue.makeCommandBuffer())
            drain.commit()
            drain.waitUntilCompleted()
            let total = CACurrentMediaTime() - start
            _ = last

            let sorted = submitMilliseconds.sorted()
            print(
                "maxFramesInFlight=\(maxFramesInFlight): \(String(format: "%.1f", Double(frameTotal) / total)) FPS; "
                    + "CPU submit ms median=\(String(format: "%.2f", sorted[sorted.count / 2])) "
                    + "p95=\(String(format: "%.2f", sorted[Int(Double(sorted.count) * 0.95)]))"
            )
        }
        Thread.sleep(forTimeInterval: 0.5)
        print("  CPU ms inside submit: " + cpu.summary(
            ["snapshot", "frameBuffers", "imageEncoder", "memoryAttention", "decoder", "selector", "memoryEncoder"]
        ))
    }
}

/// Opt-in leak probe: separates Metal's own buffer caching from retention inside
/// the package. Each block prints the footprint before, after the work, and
/// after everything is released.
@Test(.enabled(if: benchmarkEnabled))
func memoryLeakProbe() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let inputBuffers = try benchmarkInputBuffers(device: device)
    let prompts: [EfficientTAMPrompt] = [
        .init(x: 160, y: 300, label: .positivePoint),
        .init(x: 0, y: 0, label: .padding),
    ]
    func drain()
    {
        if let buffer = commandQueue.makeCommandBuffer()
        {
            buffer.commit()
            buffer.waitUntilCompleted()
        }
    }
    func report(_ label: String, before: Double, during: Double)
    {
        Thread.sleep(forTimeInterval: 1)
        let after = physicalFootprintMB()
        print(
            "\(label): before \(Int(before)) MB, after work \(Int(during)) MB, "
                + "after release \(Int(after)) MB (retained \(Int(after - before)) MB)"
        )
    }

    // 1. Control: 200 fresh 7 MB private buffers, each blit-filled, no package code.
    do
    {
        let before = physicalFootprintMB()
        for _ in 0..<200
        {
            autoreleasepool
            {
                guard let buffer = device.makeBuffer(length: 7_340_032, options: .storageModePrivate),
                      let commandBuffer = commandQueue.makeCommandBuffer(),
                      let blit = commandBuffer.makeBlitCommandEncoder() else { return }
                blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
                blit.endEncoding()
                commandBuffer.commit()
            }
        }
        drain()
        report("1 raw Metal buffers (200 x 7 MB)", before: before, during: physicalFootprintMB())
    }

    // 2. Image encoder variants on one long-lived, warmed encoder, so its own
    //    constant memory is excluded and only per-encode retention shows.
    do
    {
        let encoder = try EfficientTAMImageEncoder(commandQueue: commandQueue, maxFramesInFlight: 3)
        func encodeMany(_ count: Int, queue: MTLCommandQueue, output: () -> MTLBuffer?) throws
        {
            var done = 0
            while done < count
            {
                guard let out = output(), let commandBuffer = queue.makeCommandBuffer() else { break }
                if try encoder.encode(
                    inputBuffer: inputBuffers[done % inputBuffers.count],
                    outputBuffer: out,
                    commandBuffer: commandBuffer,
                    commit: true
                )
                {
                    done += 1
                }
            }
        }
        try encodeMany(3, queue: commandQueue) { device.makeBuffer(length: encoder.outputBufferLength, options: .storageModePrivate) }
        drain()

        func variant(_ label: String, _ work: () throws -> Void) throws
        {
            let before = physicalFootprintMB()
            try work()
            drain()
            Thread.sleep(forTimeInterval: 0.5)
            let after = physicalFootprintMB()
            print("2\(label): +\(Int(after - before)) MB over 100 encodes (\(String(format: "%.2f", (after - before) / 100)) MB each)")
        }
        try variant("a fresh output buffer per encode")
        {
            try encodeMany(100, queue: commandQueue) { device.makeBuffer(length: encoder.outputBufferLength, options: .storageModePrivate) }
        }
        let reused = device.makeBuffer(length: encoder.outputBufferLength, options: .storageModePrivate)
        try variant("b one reused output buffer")
        {
            try encodeMany(100, queue: commandQueue) { reused }
        }
        try variant("c submit path (MPSGraph allocates results)")
        {
            var done = 0
            while done < 100
            {
                guard let commandBuffer = commandQueue.makeCommandBuffer() else { break }
                if try encoder.submit(inputBuffer: inputBuffers[done % 4], commandBuffer: commandBuffer, commit: true, completion: { _ in })
                {
                    done += 1
                }
            }
        }
        try variant("d fresh output buffer, private queue")
        {
            if let privateQueue = device.makeCommandQueue()
            {
                try encodeMany(100, queue: privateQueue) { device.makeBuffer(length: encoder.outputBufferLength, options: .storageModePrivate) }
                if let waitBuffer = privateQueue.makeCommandBuffer()
                {
                    waitBuffer.commit()
                    waitBuffer.waitUntilCompleted()
                }
            }
        }
    }

    // 3. Tracker alone: footprint sampled while it is alive.
    do
    {
        let before = physicalFootprintMB()
        var during = 0.0
        do
        {
            let tracker = try EfficientTAMVideoTracker(commandQueue: commandQueue, maxFramesInFlight: 3)
            try tracker.prewarmMemoryAttention()
            var last = try #require(try tracker.encodeInitialFrame(inputBuffer: inputBuffers[0], prompts: prompts))
            var accepted = 1
            var samples: [String] = []
            while accepted < 240
            {
                if let output = try tracker.encodeNextFrame(inputBuffer: inputBuffers[accepted % inputBuffers.count])
                {
                    last = output
                    accepted += 1
                    if accepted % 40 == 0
                    {
                        drain()
                        samples.append("\(accepted): \(Int(physicalFootprintMB()))")
                    }
                }
            }
            _ = last
            print("3 tracker alive, footprint MB by frame count (drained each sample): " + samples.joined(separator: ", "))
            during = physicalFootprintMB()
        }
        report("3 tracker alone (240 frames)", before: before, during: during)
    }
}

private final class StageTimings
{
    private let lock = NSLock()
    private var totals: [String: (sum: Double, count: Int)] = [:]

    func add(_ stage: String, _ milliseconds: Double)
    {
        self.lock.lock()
        let current = self.totals[stage] ?? (0, 0)
        self.totals[stage] = (current.sum + milliseconds, current.count + 1)
        self.lock.unlock()
    }

    func summary(_ order: [String]) -> String
    {
        self.lock.lock()
        defer { self.lock.unlock() }
        var parts: [String] = []
        var sum = 0.0
        for stage in order
        {
            guard let total = self.totals[stage] else { continue }
            let mean = total.sum / Double(total.count)
            sum += mean
            parts.append("\(stage) \(String(format: "%.2f", mean))")
        }
        return parts.joined(separator: ", ") + (order.count > 1 ? "; sum \(String(format: "%.2f", sum))" : "")
    }
}

private func benchmarkInputBuffers(device: MTLDevice) throws -> [MTLBuffer]
{
    let url = try #require(
        Bundle.module.url(forResource: "tracker_frames_rgb_uint8", withExtension: "bin", subdirectory: "Fixtures")
    )
    let frames = [UInt8](try Data(contentsOf: url))
    let frameLength = 512 * 512 * 3
    return try (0..<4).map
    {
        index in
        let rgb = frames[(index * frameLength)..<((index + 1) * frameLength)].map { Float($0) / 255 }
        return try #require(device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride))
    }
}

private func physicalFootprintMB() -> Double
{
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info)
    {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count))
        {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}
