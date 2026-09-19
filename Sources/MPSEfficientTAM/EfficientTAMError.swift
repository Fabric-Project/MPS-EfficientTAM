import Foundation

public struct EfficientTAMError: LocalizedError, Sendable
{
    public let errorDescription: String?

    public init(_ message: String)
    {
        self.errorDescription = message
    }
}
