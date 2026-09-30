import MetalPerformanceShadersGraph

/// The precision an EfficientTAM stage computes in. Inputs and outputs are
/// float32 in every mode.
public enum EfficientTAMPrecision: Sendable, Equatable
{
    /// Float32 throughout.
    case float32
    /// Convolutions, linear layers and attention matmuls in float16 (weights
    /// and arithmetic), everything else float32.
    case mixedFloat16
    /// Float16 throughout, with one cast after each input and one before each
    /// output.
    case float16

    /// The type activations flow through between layers.
    var activationDataType: MPSDataType { self == .float16 ? .float16 : .float32 }

    /// The type convolutions, linear layers and attention matmuls run in.
    var layerDataType: MPSDataType { self == .float32 ? .float32 : .float16 }
}
