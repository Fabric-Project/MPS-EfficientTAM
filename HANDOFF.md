# MPS-EfficientTAM Video Tracking Handoff

Date: 2026-09-19

## Goal and constraints

Build a self-contained Swift 5.9 SPM implementation of EfficientTAM Tiny 512 in pure MPSGraph for macOS 15, iOS 18, and visionOS 2. Fabric-specific guided filtering and joint-bilateral upsampling stay outside this package. Public learned stages use the Fabric MPS-family `run` / `submit` / `encode` pattern, preserve GPU-resident buffers, and avoid CPU/GPU waits in the real-time `encode` path. Accuracy takes priority over speculative FP16 conversion.

The final repository is `/Users/vade/Documents/Repositories/Fabric/MPS-EfficientTAM`. Because that sibling is outside the writable sandbox, active edits were made with `apply_patch` in `/private/tmp/MPS-EfficientTAM-work` and then copied to the repository. At the time this handoff was written, the newest video changes are still in the staging directory and must be synchronized to the final repository.

## Verified implementation

The existing image path is complete and already present in the final repository:

- `EfficientTAMImageEncoder`: official Tiny 512 image encoder, float32, output `[1,256,32,32]` including `no_mem_embed`.
- `EfficientTAMPromptDecoder`: point/box prompting and three 128x128 mask candidates.
- `EfficientTAMMaskPromptEncoder`: iterative 128x128 prior-mask prompting.
- `EfficientTAMMaskPostprocessor`: official bilinear resizing, unthresholded raw logits.
- Full PyTorch differential fixtures and no-wait GPU-chain tests.

The staging directory adds the video primitives:

- `EfficientTAMPromptDecoder` now also returns/encodes:
  - raw object-presence logit;
  - three projected 256-value object pointers aligned with the three mask candidates;
  - `no_obj_ptr` substitution when the object is absent;
  - `NO_OBJ_SCORE` (`-1024`) mask suppression when object score is nonpositive.
- `EfficientTAMMaskSelector`: pure MPSGraph highest-IoU selection that keeps mask, IoU, and object pointer aligned without readback.
- `EfficientTAMMemoryEncoder`: official predicted-mask preprocessing, four-stage mask downsampler, two ConvNeXt fusion blocks, and 64-channel spatial-memory projection. It subtracts `no_mem_embed` internally because the existing public image embedding includes it while upstream memory encoding consumes the raw visual feature.
- `EfficientTAMMemoryAttention`: official four-layer memory transformer with exact axial RoPE, spatial memory positions, temporal memory positions, and 256-to-four-64-channel object-pointer token splitting. It compiles for exact memory-frame and pointer counts.
- `EfficientTAMVideoTracker`: first-pass single-object, forward-only, GPU-resident coordinator. It queues image encode → memory attention → tracking decode → mask selection → new-memory encode with separate ordered command buffers and no completion wait. It retains one conditioning memory, six recent spatial memories, and up to sixteen pointers.
- Weight export now writes all 455 checkpoint tensors. Native float32 weights are 71,465,096 bytes (about 68 MiB), versus the earlier 40 MiB image-only subset.

## Accuracy verified on Apple M1 Max

All 15 staging tests pass with `swift test`.

- Image encoder: existing full 262,144-value PyTorch comparison passes.
- Base decoder mask: MAE `6.2743675e-06`, max `0.00011444092`.
- Decoder IoU: max `2.3841858e-07`.
- Iterative decoder mask: MAE `1.543839e-05`, max `0.00017166138`.
- Iterative IoU: max `4.61936e-06`.
- Mask-prompt embedding: MAE `3.603446e-08`, max `4.172325e-07`.
- Official bilinear mask resize: MAE `1.9932854e-07`, max `3.8146973e-06`.
- Memory encoder: MAE `2.0966696e-07`, max `1.9073486e-06` over all 65,536 outputs.
- Memory spatial position encoding matches PyTorch within `1e-5`.
- One-memory/one-pointer memory attention: MAE `3.4441888e-07`, max `7.4505806e-06` over all 262,144 outputs.
- GPU mask selector test confirms selected mask and pointer remain aligned.

The temporary benchmark test in staging reports variable image-encoder throughput around 123–165 serialized FPS and 143–154 pipelined FPS on M1 Max. `EfficientTAMBenchmarkTests.swift` is intentionally temporary and must not be copied to the final repository.

## New/changed staging files to synchronize

Copy these from `/private/tmp/MPS-EfficientTAM-work` into matching paths under `/Users/vade/Documents/Repositories/Fabric/MPS-EfficientTAM`:

- `Sources/MPSEfficientTAM/EfficientTAMPromptDecoder.swift`
- `Sources/MPSEfficientTAM/EfficientTAMMaskSelector.swift`
- `Sources/MPSEfficientTAM/EfficientTAMMemoryEncoder.swift`
- `Sources/MPSEfficientTAM/EfficientTAMMemoryAttention.swift`
- `Sources/MPSEfficientTAM/EfficientTAMVideoTracker.swift`
- `Sources/MPSEfficientTAM/Models/EfficientTAMTiny512_weights.bin`
- `Sources/MPSEfficientTAM/Models/EfficientTAMTiny512_weights.json`
- `Tests/MPSEfficientTAMTests/EfficientTAMPromptDecoderTests.swift`
- `Tests/MPSEfficientTAMTests/EfficientTAMMemoryEncoderTests.swift`
- `Tests/MPSEfficientTAMTests/EfficientTAMMemoryAttentionTests.swift`
- all newly generated fixtures listed below;
- `Tools/export_weights.py`
- `Tools/generate_decoder_fixture.py`
- `Tools/generate_memory_encoder_fixture.py`
- `Tools/generate_memory_attention_fixture.py`
- this `HANDOFF.md`.

New fixtures:

- `decoder_object_score_reference.bin`
- `decoder_object_pointers_reference.bin`
- `memory_selected_mask_reference.bin`
- `memory_features_reference.bin`
- `memory_position_reference.bin`
- `attention_current_embedding_reference.bin`
- `attention_memory_features_reference.bin`
- `attention_memory_position_reference.bin`
- `attention_object_pointer_reference.bin`
- `attention_output_reference.bin`

Do not copy `Tests/MPSEfficientTAMTests/EfficientTAMBenchmarkTests.swift`.

## Critical next validation

`EfficientTAMVideoTracker.swift` compiles but has no end-to-end tracker test yet. This is the first task for the next model.

1. Generate a two- or three-frame upstream fixture using the official `track_step`/video predictor path, including the initial point prompt and the next frame with no prompt.
2. Add a tracker test that calls `encodeInitialFrame`, immediately calls `encodeNextFrame` on the same queue without waiting, then performs one final blit/readback and compares the selected next-frame mask, object score, pointer, and new memory to PyTorch.
3. Validate temporal indexes and memory ordering for frame counts 1 through 7. Only the one-memory/one-pointer attention shape is currently accuracy-tested.
4. Exercise at least the 2-memory/2-pointer and 7-memory/7-pointer graph shapes. The RoPE implementation repeats the same 32x32 frequency block per spatial memory as upstream.

## Known risks in the first tracker coordinator

- It is single-object and forward-only. There is no correction-prompt API on later frames, reverse propagation, object removal, or multi-object non-overlap handling yet.
- The coordinator queues work without waiting, but its long `guard` chain commits earlier stages before later-stage backpressure is known. The tracker-level semaphore should keep corresponding stage slots available, but this should be made transactional or explicitly stress-tested.
- `memoryAttention(memoryCount:pointerCount:)` currently creates attention instances with `maxFramesInFlight: 3` instead of preserving the tracker initializer’s value.
- Attention executables are lazily compiled for each exact `(memoryCount, pointerCount)` pair. This preserves speed and avoids always attending to padded memories, but can cause a first-use compile hitch and retains multiple copies of graph constants. Add a prewarm API and measure memory before deciding whether to keep this strategy.
- `EfficientTAMVideoTrackingOutput` returns private GPU buffers and no completion callback. Same-queue downstream work is safe through Metal queue ordering; clients needing CPU access must schedule their own completion/readback.
- The tracker currently treats the first conditioning frame as the sole conditioning memory. Upstream supports multiple corrected conditioning frames.
- Temporal position selection is implemented as conditioning index `6` and recent non-conditioning indexes `0...5` from newest to oldest. Confirm this with a multi-frame PyTorch fixture.
- Memory encoder currently accepts selected 128x128 predicted logits and internally performs official 512x512 resize plus `sigmoid * 20 - 10`. Direct user mask inputs follow a separate upstream path and are not yet supported by the tracker.
- Decoder single-mask token mode is not implemented. Current tracking continues to produce three candidates and selects best IoU, matching the configured Tiny model’s multimask tracking behavior for zero/one-point frames, but correction cases with more points need upstream mode parity.
- Object-absence behavior was added after the first decoder fixtures. Existing positive-object tests pass, but an explicit occluded/no-object fixture is still required.
- README has not yet been updated for the new video types.
- iOS and visionOS builds remain unverified.

## Recommended implementation order

1. Sync staging to the final repository, run `swift test`, and confirm Git LFS tracks the enlarged `.bin` (`.gitattributes` already contains `*.bin filter=lfs`).
2. Add the two-frame tracker differential fixture and no-wait test.
3. Fix any temporal ordering or tracker orchestration errors discovered by that test.
4. Add tracker correction prompts and direct mask conditioning frames.
5. Add reverse tracking and multiple conditioning frames.
6. Add multi-object state, object IDs, and optional non-overlap constraint as a separate composable stage.
7. Benchmark image encoder, memory attention at 1/3/7 memories, decoder, and full frame-to-frame throughput. Keep float32 until accuracy baselines are locked.
8. Update README, run debug/release plus iOS/visionOS builds, then commit/tag. The final repository is still mostly untracked from the empty-repository starting point.

## Useful commands and resources

```sh
cd /private/tmp/MPS-EfficientTAM-work
swift test
swift test -c release

/private/tmp/efficienttam-venv/bin/python Tools/export_weights.py \
  /private/tmp/efficienttam_ti_512x512.pt \
  Sources/MPSEfficientTAM/Models/EfficientTAMTiny512_weights.bin \
  Sources/MPSEfficientTAM/Models/EfficientTAMTiny512_weights.json
```

Official upstream checkout: `/private/tmp/EfficientTAM-upstream`

Official checkpoint: `/private/tmp/efficienttam_ti_512x512.pt`

Python environment: `/private/tmp/efficienttam-venv`

Final repository: `/Users/vade/Documents/Repositories/Fabric/MPS-EfficientTAM`

Verified staging directory: `/private/tmp/MPS-EfficientTAM-work`
