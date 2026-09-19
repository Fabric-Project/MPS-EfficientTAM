import Foundation
import Metal
import MetalPerformanceShaders

/// How every stage encodes onto a command buffer.
///
/// Fabric's per-frame buffer is a single long-lived `MPSCommandBuffer`.
/// `MPSGraphExecutable.encode` may `commitAndContinue` it, which commits the
/// underlying `MTLCommandBuffer` and swaps in a new one inside the same
/// `MPSCommandBuffer`. So when the caller already hands us an
/// `MPSCommandBuffer`, stages must encode onto that exact instance and never
/// wrap it again: a second wrapper doesn't know about the swap, so anything it
/// commits or observes can be the wrong underlying buffer.
///
/// A plain `MTLCommandBuffer` is still accepted. It is wrapped here, which is
/// only safe when this call is also the one that commits it.
enum EfficientTAMCommandBuffer
{
    struct Target
    {
        /// The buffer to encode onto.
        let commandBuffer: MPSCommandBuffer
        /// What the caller passed in.
        let original: MTLCommandBuffer
        /// True when the caller already provided an `MPSCommandBuffer`.
        let callerOwnsWrapper: Bool
    }

    static func target(for commandBuffer: MTLCommandBuffer) -> Target
    {
        if let existing = commandBuffer as? MPSCommandBuffer
        {
            return Target(commandBuffer: existing, original: commandBuffer, callerOwnsWrapper: true)
        }
        return Target(
            commandBuffer: MPSCommandBuffer(commandBuffer: commandBuffer),
            original: commandBuffer,
            callerOwnsWrapper: false
        )
    }

    /// The tail of every `encode`, run after the executables have been encoded.
    ///
    /// The completion handler goes on the live underlying buffer, so after a
    /// split it fires when the last segment completes rather than the first.
    /// It is registered before any commit, which Metal requires. Throws, having
    /// registered nothing, when a plain buffer was split but the caller asked
    /// to keep it open: its remaining work would be stranded in a wrapper the
    /// caller cannot reach.
    static func finish(_ target: Target, commit: Bool, onCompletion: @escaping () -> Void) throws
    {
        if !target.callerOwnsWrapper, !commit, target.original.status != .notEnqueued
        {
            throw EfficientTAMError(
                "MPSGraph committed the command buffer partway through encoding, so it cannot be left open with commit: false. "
                    + "Pass commit: true, or encode onto an MPSCommandBuffer such as Fabric's frame buffer."
            )
        }
        target.commandBuffer.addCompletedHandler { _ in onCompletion() }
        if commit
        {
            autoreleasepool { target.commandBuffer.commit() }
        }
    }
}
