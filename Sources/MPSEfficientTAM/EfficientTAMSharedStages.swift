import Foundation
import Metal

/// The tracker's stateless stages, built once per device and shared by every
/// `EfficientTAMVideoTracker` on it. A tracker's memory bank, frame counter and
/// prompt buffers are its own; the compiled graphs and the weights inside them
/// are identical for every tracker, so a second tracker costs only its memory
/// bank instead of another image encoder, decoder, memory encoder and attention
/// graph.
///
/// Re-entrant by construction: lookup and creation happen under one lock, so
/// two trackers created at once get the same instance. Sharing a stage shares
/// no per-frame state, since every `encode` takes its inputs and outputs from
/// the caller. The one shared resource is in-flight capacity: a stage slot is
/// held from encode until its command buffer completes, so the stages are sized
/// for many trackers (`framesInFlight`), and each tracker still applies its own
/// admission limit on top.
///
/// Held weakly by the cache, so the stages are released when the last tracker
/// using them goes away.
final class EfficientTAMSharedStages
{
    /// In-flight frames each shared stage allows, summed over every tracker on
    /// this device. A tracker asks for `maxFramesInFlight` of these, so the
    /// default of 3 supports five trackers at once. Stage slots cost little
    /// until used.
    static let framesInFlight = 16

    /// The decoder is compiled for one point plus one padding token, which is
    /// also exactly what an unprompted tracking frame uses, so one decoder
    /// serves both.
    static let promptTokenCount = 2

    let imageEncoder: EfficientTAMImageEncoder
    let promptDecoder: EfficientTAMPromptDecoder
    let maskSelector: EfficientTAMMaskSelector
    let memoryEncoder: EfficientTAMMemoryEncoder

    private let commandQueue: MTLCommandQueue
    private let precision: EfficientTAMPrecision
    private let computeUnits: EfficientTAMComputeUnits
    private let attentionLock = NSLock()
    private var maskedAttention: EfficientTAMMemoryAttention?

    private final class WeakStages
    {
        weak var stages: EfficientTAMSharedStages?
        init(_ stages: EfficientTAMSharedStages) { self.stages = stages }
    }

    /// One set of stages per device, precision and compute units: trackers
    /// share compiled graphs only when they were built the same way.
    private struct CacheKey: Hashable
    {
        let deviceIdentifier: UInt64
        let precision: EfficientTAMPrecision
        let computeUnits: EfficientTAMComputeUnits
    }

    private static let cacheLock = NSLock()
    private static var cache: [CacheKey: WeakStages] = [:]

    static func stages(commandQueue: MTLCommandQueue, precision: EfficientTAMPrecision, computeUnits: EfficientTAMComputeUnits) throws -> EfficientTAMSharedStages
    {
        let key = CacheKey(deviceIdentifier: commandQueue.device.registryID, precision: precision, computeUnits: computeUnits)
        self.cacheLock.lock()
        defer { self.cacheLock.unlock() }

        self.cache = self.cache.filter { $0.value.stages != nil }
        if let existing = self.cache[key]?.stages { return existing }
        let created = try EfficientTAMSharedStages(commandQueue: commandQueue, precision: precision, computeUnits: computeUnits)
        self.cache[key] = WeakStages(created)
        return created
    }

    private init(commandQueue: MTLCommandQueue, precision: EfficientTAMPrecision, computeUnits: EfficientTAMComputeUnits) throws
    {
        self.commandQueue = commandQueue
        self.precision = precision
        self.computeUnits = computeUnits
        self.imageEncoder = try EfficientTAMImageEncoder(
            commandQueue: commandQueue,
            maxFramesInFlight: Self.framesInFlight,
            precision: precision,
            computeUnits: computeUnits
        )
        self.promptDecoder = try EfficientTAMPromptDecoder(
            promptCount: Self.promptTokenCount,
            commandQueue: commandQueue,
            maxFramesInFlight: Self.framesInFlight,
            precision: precision,
            computeUnits: computeUnits
        )
        self.maskSelector = try EfficientTAMMaskSelector(
            commandQueue: commandQueue,
            maxFramesInFlight: Self.framesInFlight,
            computeUnits: computeUnits
        )
        self.memoryEncoder = try EfficientTAMMemoryEncoder(
            commandQueue: commandQueue,
            maxFramesInFlight: Self.framesInFlight,
            precision: precision,
            computeUnits: computeUnits
        )
    }

    /// The one masked memory-attention graph, compiled on first use (about a
    /// second) and shared. Compiling holds a lock, so trackers asking at the
    /// same moment wait for the first's compile and then get the same graph.
    func maskedMemoryAttention() throws -> EfficientTAMMemoryAttention
    {
        self.attentionLock.lock()
        defer { self.attentionLock.unlock() }
        if let maskedAttention = self.maskedAttention { return maskedAttention }
        let attention = try EfficientTAMMemoryAttention(
            memoryFrameCount: EfficientTAMVideoTracker.maximumSpatialMemoryCount,
            objectPointerCount: EfficientTAMVideoTracker.maximumObjectPointerCount,
            usesKeyMask: true,
            commandQueue: self.commandQueue,
            maxFramesInFlight: Self.framesInFlight,
            precision: self.precision,
            computeUnits: self.computeUnits
        )
        self.maskedAttention = attention
        return attention
    }
}
