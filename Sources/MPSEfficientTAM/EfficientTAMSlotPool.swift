import Foundation

/// A fixed pool of in-flight slots. Unlike `DispatchSemaphore`, which traps if
/// it is deallocated while its value is below its initial count, this can be
/// released while GPU work is still in flight, for example when a client tears
/// down a node mid-frame. Completion handlers hold their owner weakly, so
/// slots simply stop being returned once the owner is gone.
final class EfficientTAMSlotPool
{
    private let condition = NSCondition()
    private var freeSlots: [Int]

    init(count: Int)
    {
        self.freeSlots = Array(0..<count)
    }

    /// Returns a free slot immediately, or nil if every slot is in flight.
    func tryAcquire() -> Int?
    {
        self.condition.lock()
        defer { self.condition.unlock() }
        return self.freeSlots.popLast()
    }

    /// Blocks until a slot is free.
    func acquire() -> Int
    {
        self.condition.lock()
        defer { self.condition.unlock() }
        while self.freeSlots.isEmpty
        {
            self.condition.wait()
        }
        return self.freeSlots.removeLast()
    }

    func release(_ slot: Int)
    {
        self.condition.lock()
        self.freeSlots.append(slot)
        self.condition.unlock()
        self.condition.signal()
    }
}
