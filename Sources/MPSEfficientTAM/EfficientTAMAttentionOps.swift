import Foundation
import MetalPerformanceShadersGraph

/// Shared attention primitive, `softmax(scale * Q K^T + mask) V`. By default it
/// emits the explicit matmul/softmax/matmul sequence. MPSGraph's fused
/// scaled-dot-product attention is available by setting
/// `EFFICIENTTAM_FUSED_ATTENTION`, but on an M1 Max it measured slower than the
/// explicit sequence (image encoder +18%, decoder 2x, memory attention +13%).
enum EfficientTAMAttentionOps
{
    static let usesFusedAttention = ProcessInfo.processInfo.environment["EFFICIENTTAM_FUSED_ATTENTION"] != nil

    /// `query` is `[B, H, Nq, F]`, `key` and `value` are `[B, H, Nkv, F]`. The
    /// optional additive `mask` must broadcast against `[B, H, Nq, Nkv]`.
    /// The two matmuls run in `precision.layerDataType`; the scale, mask and
    /// softmax in `precision.activationDataType`, which is also the result's
    /// type. At float32 no casts are added.
    static func attention(
        graph: MPSGraph,
        query: MPSGraphTensor,
        key: MPSGraphTensor,
        value: MPSGraphTensor,
        mask: MPSGraphTensor?,
        scale: Float,
        precision: EfficientTAMPrecision = .float32
    ) -> MPSGraphTensor
    {
        let layerType = precision.layerDataType
        let activationType = precision.activationDataType
        func cast(_ tensor: MPSGraphTensor, to dataType: MPSDataType) -> MPSGraphTensor
        {
            tensor.dataType == dataType ? tensor : graph.cast(tensor, to: dataType, name: nil)
        }
        if self.usesFusedAttention
        {
            return cast(graph.scaledDotProductAttention(
                query: cast(query, to: layerType),
                key: cast(key, to: layerType),
                value: cast(value, to: layerType),
                mask: mask.map { cast($0, to: layerType) },
                scale: scale,
                name: nil
            ), to: activationType)
        }
        let transposedKey = graph.transpose(cast(key, to: layerType), permutation: [0, 1, 3, 2], name: nil)
        var scores = cast(graph.matrixMultiplication(primary: cast(query, to: layerType), secondary: transposedKey, name: nil), to: activationType)
        scores = graph.multiplication(scores, graph.constant(Double(scale), dataType: activationType), name: nil)
        if let mask
        {
            scores = graph.addition(scores, cast(mask, to: activationType), name: nil)
        }
        let probabilities = graph.softMax(with: scores, axis: 3, name: nil)
        return cast(graph.matrixMultiplication(primary: cast(probabilities, to: layerType), secondary: cast(value, to: layerType), name: nil), to: activationType)
    }
}
