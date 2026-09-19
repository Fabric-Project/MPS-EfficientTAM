#!/usr/bin/env python3
"""Generate a multi-frame official-PyTorch video tracking fixture.

Runs the upstream `track_step` path (memory selection, temporal position
indexes, object pointers, memory encoding) over real video frames with one
positive point on the first frame and no prompts afterwards. One frame is
blacked out so the object-absent path (mask suppression, `no_obj_ptr`) runs and
the following frame must recover from an absent-object memory. State is kept in
float32; the official video predictor stores `maskmem_features` as bfloat16,
which this fixture intentionally does not reproduce.

Outputs (little-endian):
  tracker_frames_rgb_uint8.bin        [F,512,512,3] uint8, consumed by the test
  tracker_masks_reference.bin         [F,128,128]   selected mask logits
  tracker_iou_reference.bin           [F]           selected IoU prediction
  tracker_object_score_reference.bin  [F]           raw object-presence logit
  tracker_object_pointer_reference.bin[F,256]       selected object pointer
  tracker_memory_features_reference.bin [F,64,32,32] new spatial memory
"""

import argparse
import pathlib
import subprocess
import sys

import numpy as np
import torch

FRAME_COUNT = 11
OCCLUDED_FRAME = 9  # replaced by a black frame to exercise the object-absent path
FRAME_STRIDE = 3
POINT = (160.0, 300.0)


def load_frames(video: pathlib.Path) -> np.ndarray:
    command = [
        "ffmpeg", "-loglevel", "error", "-i", str(video),
        "-vf", f"select='not(mod(n,{FRAME_STRIDE}))',scale=512:512:flags=bilinear",
        "-frames:v", str(FRAME_COUNT), "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 512, 512, 3)
    assert frames.shape[0] == FRAME_COUNT, frames.shape
    frames = frames.copy()
    frames[OCCLUDED_FRAME] = 0
    return frames


def normalized(frame: np.ndarray) -> torch.Tensor:
    rgb = torch.from_numpy(frame.copy()).float().reshape(1, 512, 512, 3) / 255.0
    rgb = (rgb - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    return rgb.permute(0, 3, 1, 2)


def write(directory: pathlib.Path, name: str, array: np.ndarray) -> None:
    (directory / name).write_bytes(np.ascontiguousarray(array).tobytes(order="C"))
    print(f"{name}: shape={array.shape} dtype={array.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream_checkout", type=pathlib.Path)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("video", type=pathlib.Path)
    parser.add_argument("output_directory", type=pathlib.Path)
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.upstream_checkout))
    torch.mps.is_available = lambda: True
    from efficient_track_anything.build_efficienttam import build_efficienttam

    model = build_efficienttam(
        "configs/efficienttam/efficienttam_ti_512x512.yaml",
        str(arguments.checkpoint),
        device="cpu",
        hydra_overrides_extra=["++model.compile_image_encoder=False"],
        apply_postprocessing=False,
    )

    captured = {}
    original_heads = model._forward_sam_heads

    def capturing_heads(*args, **kwargs):
        result = original_heads(*args, **kwargs)
        captured["ious"] = result[2]
        return result

    model._forward_sam_heads = capturing_heads

    frames = load_frames(arguments.video)
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    masks, ious, scores, pointers, memories = [], [], [], [], []

    with torch.no_grad():
        for index in range(FRAME_COUNT):
            backbone_out = model.forward_image(normalized(frames[index]))
            _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
            is_initial = index == 0
            point_inputs = None
            if is_initial:
                point_inputs = {
                    "point_coords": torch.tensor([[list(POINT)]], dtype=torch.float32),
                    "point_labels": torch.tensor([[1]], dtype=torch.int32),
                }
            current = model.track_step(
                frame_idx=index,
                is_init_cond_frame=is_initial,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=FRAME_COUNT,
            )
            key = "cond_frame_outputs" if is_initial else "non_cond_frame_outputs"
            output_dict[key][index] = current

            selected_iou = captured["ious"].max(dim=-1).values
            masks.append(current["pred_masks"][0, 0])
            ious.append(selected_iou[0])
            scores.append(current["object_score_logits"][0, 0])
            pointers.append(current["obj_ptr"][0])
            memories.append(current["maskmem_features"][0])
            print(
                f"frame {index}: score={scores[-1].item():+.4f} iou={ious[-1].item():.4f} "
                f"mask>0={(masks[-1] > 0).float().mean().item():.4f} "
                f"mask range=({masks[-1].min().item():.2f},{masks[-1].max().item():.2f})"
            )

    def stack(values):
        return torch.stack(values).detach().float().contiguous().numpy().astype("<f4")

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    write(arguments.output_directory, "tracker_frames_rgb_uint8.bin", frames)
    write(arguments.output_directory, "tracker_masks_reference.bin", stack(masks))
    write(arguments.output_directory, "tracker_iou_reference.bin", stack(ious))
    write(arguments.output_directory, "tracker_object_score_reference.bin", stack(scores))
    write(arguments.output_directory, "tracker_object_pointer_reference.bin", stack(pointers))
    write(arguments.output_directory, "tracker_memory_features_reference.bin", stack(memories))


if __name__ == "__main__":
    main()
