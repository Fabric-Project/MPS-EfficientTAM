import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Testing
@testable import MPSEfficientTAM

/// The precision tier under test, from `EFFICIENTTAM_PRECISION`
/// (`float32`, `mixedFloat16` or `float16`; float32 when unset).
func efficientTAMTestPrecision() -> EfficientTAMPrecision
{
    switch ProcessInfo.processInfo.environment["EFFICIENTTAM_PRECISION"]
    {
    case "mixedFloat16": .mixedFloat16
    case "float16": .float16
    default: .float32
    }
}

/// Peak signal-to-noise ratio of `candidate` against `reference`, using the
/// reference's own peak magnitude.
func efficientTAMPSNR(_ candidate: [Float], _ reference: [Float]) -> Double
{
    let meanSquaredError = zip(candidate, reference).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) } / Double(max(reference.count, 1))
    let peak = Double(reference.map { abs($0) }.max() ?? 0)
    guard meanSquaredError > 0 else { return .infinity }
    return 10 * log10(peak * peak / meanSquaredError)
}

/// A real 512×512 tracker frame (the first of the fixture clip) as NHWC
/// float32 RGB in 0...1.
func efficientTAMRealFrame(device: MTLDevice) throws -> MTLBuffer
{
    let url = try #require(Bundle.module.url(forResource: "tracker_frames_rgb_uint8", withExtension: "bin", subdirectory: "Fixtures"))
    let frameLength = EfficientTAMImageEncoder.inputWidth * EfficientTAMImageEncoder.inputHeight * 3
    let rgb = try Data(contentsOf: url).prefix(frameLength).map { Float($0) / 255 }
    return try #require(device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride))
}

/// The reduced-precision image encoder against the float32 one on a real
/// frame. Gate: 40 dB, the CoreAI model-authoring threshold for a float16
/// build against its reference. Skipped for float32.
///
/// `EFFICIENTTAM_PRECISION=float16 swift test --filter imageEncoderPrecisionMatchesFloat32OnRealFrame`
@Test func imageEncoderPrecisionMatchesFloat32OnRealFrame() throws
{
    let precision = efficientTAMTestPrecision()
    guard precision != .float32,
          let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let frame = try efficientTAMRealFrame(device: device)
    let reference = try EfficientTAMImageEncoder(commandQueue: commandQueue, maxFramesInFlight: 1).run(inputBuffer: frame)
    let candidate = try EfficientTAMImageEncoder(commandQueue: commandQueue, maxFramesInFlight: 1, precision: precision).run(inputBuffer: frame)
    let nonFiniteCount = candidate.filter { !$0.isFinite }.count
    let psnr = efficientTAMPSNR(candidate, reference)
    print("EfficientTAM image encoder \(precision) vs float32 on a real frame: \(psnr.formatted(.number.precision(.fractionLength(1)))) dB, \(nonFiniteCount) non-finite")
    #expect(nonFiniteCount == 0)
    #expect(psnr > 40)
}

/// Opt-in: finds which op MPSGraph aborts on (`bad_optional_access`) when
/// building a float16 graph. Builds one small float16 graph per candidate op,
/// printing each name to stderr (unbuffered) first: the last name printed
/// before a crash is the culprit.
///
/// `EFFICIENTTAM_RUN_FLOAT16_OP_PROBE=1 swift test --filter float16OpCompilationProbe`
@Test func float16OpCompilationProbe() throws
{
    guard ProcessInfo.processInfo.environment["EFFICIENTTAM_RUN_FLOAT16_OP_PROBE"] != nil,
          let device = MTLCreateSystemDefaultDevice() else
    {
        return
    }
    typealias Candidate = (name: String, shape: [NSNumber], body: (MPSGraph, MPSGraphTensor) -> MPSGraphTensor)
    func half(_ graph: MPSGraph, _ value: Double) -> MPSGraphTensor { graph.constant(value, dataType: .float16) }
    let candidates: [Candidate] = [
        ("erf", [1, 32, 32, 192], { graph, input in graph.erf(with: input, name: nil) }),
        ("division", [1, 3, 64, 64], { graph, input in graph.division(input, half(graph, 0.229), name: nil) }),
        ("layer norm over last axis", [1, 32, 32, 192], { graph, input in
            let mean = graph.mean(of: input, axes: [-1], name: nil)
            let variance = graph.variance(of: input, mean: mean, axes: [-1], name: nil)
            return graph.normalize(input, mean: mean, variance: variance, gamma: half(graph, 1), beta: half(graph, 0), epsilon: 1e-6, name: nil)
        }),
        ("layer norm over channels", [1, 256, 32, 32], { graph, input in
            let mean = graph.mean(of: input, axes: [1], name: nil)
            let variance = graph.variance(of: input, mean: mean, axes: [1], name: nil)
            return graph.normalize(input, mean: mean, variance: variance, gamma: half(graph, 1), beta: half(graph, 0), epsilon: 1e-6, name: nil)
        }),
        ("softmax", [1, 3, 1024, 1024], { graph, input in graph.softMax(with: input, axis: 3, name: nil) }),
        ("constant pad", [1, 32, 32, 192], { graph, input in
            graph.padTensor(input, with: .constant, leftPadding: [0, 0, 0, 0], rightPadding: [0, 10, 10, 0], constantValue: 0, name: nil)
        }),
        ("rank-6 window transpose", [1, 42, 42, 192], { graph, input in
            let windows = graph.transpose(graph.reshape(input, shape: [1, 3, 14, 3, 14, 192], name: nil), permutation: [0, 1, 3, 2, 4, 5], name: nil)
            return graph.reshape(windows, shape: [9, 14, 14, 192], name: nil)
        }),
        ("strided slice", [1, 42, 42, 192], { graph, input in
            graph.sliceTensor(input, starts: [0, 0, 0, 0], ends: [1, 32, 32, 192], strides: [1, 1, 1, 1], name: nil)
        }),
        ("rank-5 qkv transpose", [1, 1024, 3, 3, 64], { graph, input in graph.transpose(input, permutation: [2, 0, 3, 1, 4], name: nil) }),
        ("batched matmul", [1, 3, 1024, 64], { graph, input in
            graph.matrixMultiplication(primary: input, secondary: graph.transpose(input, permutation: [0, 1, 3, 2], name: nil), name: nil)
        }),
    ]
    let descriptor = MPSGraphCompilationDescriptor()
    descriptor.optimizationLevel = .level1
    descriptor.waitForCompilationCompletion = true
    for candidate in candidates
    {
        FileHandle.standardError.write(Data("float16 op probe: building \(candidate.name)\n".utf8))
        let graph = MPSGraph()
        let input = graph.placeholder(shape: candidate.shape, dataType: .float32, name: nil)
        let output = graph.cast(candidate.body(graph, graph.cast(input, to: .float16, name: nil)), to: .float32, name: nil)
        let inputType = MPSGraphShapedType(shape: candidate.shape, dataType: .float32)
        let executable = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [input: inputType],
            targetTensors: [output],
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        executable.specialize(with: MPSGraphDevice(mtlDevice: device), inputTypes: [inputType], compilationDescriptor: descriptor)
        FileHandle.standardError.write(Data("float16 op probe: \(candidate.name) OK\n".utf8))
    }
}

/// Opt-in Neural Engine check for one stage: builds it in
/// `EFFICIENTTAM_PRECISION` (float16 by default here) and runs it for
/// `EFFICIENTTAM_ANE_PROBE_SECONDS` (default 10) while powermetrics watches.
///
/// `EFFICIENTTAM_RUN_MEMORY_ENCODER_ANE_PROBE=1 swift test -c release -Xswiftc -enable-testing --filter memoryEncoderANEProbe`
@Test func memoryEncoderANEProbe() throws
{
    guard ProcessInfo.processInfo.environment["EFFICIENTTAM_RUN_MEMORY_ENCODER_ANE_PROBE"] != nil,
          let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let precision: EfficientTAMPrecision = ProcessInfo.processInfo.environment["EFFICIENTTAM_PRECISION"] == nil ? .float16 : efficientTAMTestPrecision()
    let clock = ContinuousClock()
    let compileStart = clock.now
    let encoder = try EfficientTAMMemoryEncoder(commandQueue: commandQueue, maxFramesInFlight: 1, precision: precision)
    let compileTime = clock.now - compileStart
    let imageEmbedding = try #require(device.makeBuffer(length: encoder.imageEmbeddingBufferLength, options: .storageModePrivate))
    let maskLogits = try #require(device.makeBuffer(length: encoder.maskLogitsBufferLength, options: .storageModePrivate))
    let memoryFeatures = try #require(device.makeBuffer(length: encoder.memoryFeaturesBufferLength, options: .storageModePrivate))

    let runSeconds = ProcessInfo.processInfo.environment["EFFICIENTTAM_ANE_PROBE_SECONDS"].flatMap { Int($0) } ?? 10
    print("EfficientTAM memory encoder ANE probe (\(precision)): compile \(compileTime), running \(runSeconds) s from \(Date.now.formatted(date: .omitted, time: .standard))")
    let deadline = clock.now + .seconds(runSeconds)
    var iterations = 0
    while clock.now < deadline
    {
        let commandBuffer = MPSCommandBuffer(commandBuffer: try #require(commandQueue.makeCommandBuffer()))
        _ = try encoder.encode(
            imageEmbeddingBuffer: imageEmbedding,
            maskLogitsBuffer: maskLogits,
            memoryFeaturesBuffer: memoryFeatures,
            commandBuffer: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        iterations += 1
    }
    print("EfficientTAM memory encoder ANE probe: \(iterations) runs, \((Double(runSeconds) * 1000 / Double(max(iterations, 1))).formatted(.number.precision(.fractionLength(3)))) ms/run")
}
