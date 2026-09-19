#!/usr/bin/env python3
"""Generate official PyTorch point-prompt mask and IoU fixtures."""

import argparse
import pathlib
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream_checkout", type=pathlib.Path)
    parser.add_argument("checkpoint", type=pathlib.Path)
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

    pixel_count = 512 * 512 * 3
    rgb = torch.arange(pixel_count, dtype=torch.int64).remainder(251).float() / 250.0
    rgb = rgb.reshape(1, 512, 512, 3)
    normalized = (rgb - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
        [0.229, 0.224, 0.225]
    )
    normalized = normalized.permute(0, 3, 1, 2)

    with torch.no_grad():
        image_embedding = model.forward_image(normalized)["backbone_fpn"][-1]
        image_embedding = image_embedding + model.no_mem_embed.permute(1, 2, 0).reshape(1, 256, 1, 1)
        coordinates = torch.tensor([[[160.0, 240.0]]], dtype=torch.float32)
        labels = torch.tensor([[1]], dtype=torch.int32)
        sparse, dense = model.sam_prompt_encoder(
            points=(coordinates, labels),
            boxes=None,
            masks=None,
        )
        masks, iou, sam_tokens, object_score = model.sam_mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[],
        )
        object_pointers = model.obj_ptr_proj(sam_tokens)
        object_present = (object_score > 0).float()
        object_pointers = object_present[:, :, None] * object_pointers
        object_pointers = object_pointers + (1 - object_present[:, :, None]) * model.no_obj_ptr

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    masks = masks.detach().float().contiguous().numpy().astype("<f4")
    iou = iou.detach().float().contiguous().numpy().astype("<f4")
    (arguments.output_directory / "decoder_masks_reference.bin").write_bytes(masks.tobytes(order="C"))
    (arguments.output_directory / "decoder_iou_reference.bin").write_bytes(iou.tobytes(order="C"))
    object_score = object_score.detach().float().contiguous().numpy().astype("<f4")
    object_pointers = object_pointers.detach().float().contiguous().numpy().astype("<f4")
    (arguments.output_directory / "decoder_object_score_reference.bin").write_bytes(
        object_score.tobytes(order="C")
    )
    (arguments.output_directory / "decoder_object_pointers_reference.bin").write_bytes(
        object_pointers.tobytes(order="C")
    )
    print(
        f"masks={masks.shape} mean={masks.mean():.9f} min={masks.min():.9f} max={masks.max():.9f}; "
        f"iou={iou.tolist()}"
    )


if __name__ == "__main__":
    main()
