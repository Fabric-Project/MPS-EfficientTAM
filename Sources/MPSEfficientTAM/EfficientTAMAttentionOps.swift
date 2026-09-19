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
    static func attention(
        graph: MPSGraph,
        query: MPSGraphTensor,
        key: MPSGraphTensor,
        value: MPSGraphTensor,
        mask: MPSGraphTensor?,
        scale: Float
    ) -> MPSGraphTensor
    {
        if self.usesFusedAttention
        {
            return graph.scaledDotProductAttention(
                query: query,
                key: key,
                value: value,
                mask: mask,
                scale: scale,
                name: nil
            )
        }
        let transposedKey = graph.transpose(key, permutation: [0, 1, 3, 2], name: nil)
        var scores = graph.matrixMultiplication(primary: query, secondary: transposedKey, name: nil)
        scores = graph.multiplication(scores, graph.constant(Double(scale), dataType: .float32), name: nil)
        if let mask
        {
            scores = graph.addition(scores, mask, name: nil)
        }
        let probabilities = graph.softMax(with: scores, axis: 3, name: nil)
        return graph.matrixMultiplication(primary: probabilities, secondary: value, name: nil)
    }
}
