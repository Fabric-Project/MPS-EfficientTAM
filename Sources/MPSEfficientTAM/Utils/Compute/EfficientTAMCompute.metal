//
//  EfficientTAMCompute.metal
//  MPSEfficientTAM
//
//  Packs an arbitrary source texture into the tightly-packed NHWC float32 RGB
//  buffer EfficientTAMImageEncoder requires, and writes the tracker's 128x128
//  mask logits into a texture. They are implementation details of the model's
//  tensor contract, so they live in this package rather than in a host app.
//
//  Neither kernel touches color: texels are passed through as sampled, so
//  whatever color space the host provides is the color space the model sees.
//

#include <metal_stdlib>
using namespace metal;

struct EfficientTAMPrepareUniforms
{
    uint2 outputSize;
    float4x4 textureTransform;
};

struct EfficientTAMMaskUniforms
{
    uint2 modelSize;
    uint2 outputSize;
    uint applySigmoid;
};

kernel void efficientTAMPrepareRGB(
    texture2d<float, access::sample> inputTexture [[texture(0)]],
    device float *outputRGB [[buffer(0)]],
    constant EfficientTAMPrepareUniforms &uniforms [[buffer(1)]],
    uint2 position [[thread_position_in_grid]])
{
    if (any(position >= uniforms.outputSize))
    {
        return;
    }

    constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 canonicalCoordinate = (float2(position) + 0.5) / float2(uniforms.outputSize);
    float2 storedCoordinate = (uniforms.textureTransform * float4(canonicalCoordinate, 0.0, 1.0)).xy;
    float3 rgb = inputTexture.sample(linearSampler, storedCoordinate).rgb;
    uint outputIndex = (position.y * uniforms.outputSize.x + position.x) * 3;
    outputRGB[outputIndex] = rgb.r;
    outputRGB[outputIndex + 1] = rgb.g;
    outputRGB[outputIndex + 2] = rgb.b;
}

kernel void efficientTAMWriteMask(
    device const float *modelLogits [[buffer(0)]],
    constant EfficientTAMMaskUniforms &uniforms [[buffer(1)]],
    texture2d<float, access::write> outputTexture [[texture(0)]],
    uint2 position [[thread_position_in_grid]])
{
    if (any(position >= uniforms.outputSize))
    {
        return;
    }

    // Bilinear, half-pixel-centered (align_corners = false), the same as the
    // official mask resize. At outputSize == modelSize it is the identity.
    float2 sourcePosition = (float2(position) + 0.5)
        * float2(uniforms.modelSize) / float2(uniforms.outputSize) - 0.5;
    int2 sourceMinimum = int2(floor(sourcePosition));
    float2 interpolation = fract(sourcePosition);
    uint2 maximumPosition = uniforms.modelSize - 1;
    uint2 topLeft = min(uint2(max(sourceMinimum, int2(0))), maximumPosition);
    uint2 bottomRight = min(uint2(max(sourceMinimum + 1, int2(0))), maximumPosition);
    uint2 topRight = uint2(bottomRight.x, topLeft.y);
    uint2 bottomLeft = uint2(topLeft.x, bottomRight.y);

    float topLeftLogit = modelLogits[topLeft.y * uniforms.modelSize.x + topLeft.x];
    float topRightLogit = modelLogits[topRight.y * uniforms.modelSize.x + topRight.x];
    float bottomLeftLogit = modelLogits[bottomLeft.y * uniforms.modelSize.x + bottomLeft.x];
    float bottomRightLogit = modelLogits[bottomRight.y * uniforms.modelSize.x + bottomRight.x];
    float topLogit = mix(topLeftLogit, topRightLogit, interpolation.x);
    float bottomLogit = mix(bottomLeftLogit, bottomRightLogit, interpolation.x);
    float logit = mix(topLogit, bottomLogit, interpolation.y);

    float value = uniforms.applySigmoid != 0 ? 1.0 / (1.0 + exp(-logit)) : logit;
    outputTexture.write(float4(value, value, value, 1.0), position);
}
