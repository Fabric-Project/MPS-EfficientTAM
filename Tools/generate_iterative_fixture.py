#!/usr/bin/env python3
"""Generate official PyTorch iterative-prompt and resized-logit fixtures."""

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as functional


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

    prior = torch.arange(128 * 128, dtype=torch.int64).remainder(113).float() / 29.0 - 1.5
    prior = prior.reshape(1, 1, 128, 128)
    coordinates = torch.tensor([[[160.0, 240.0]]], dtype=torch.float32)
    labels = torch.tensor([[1]], dtype=torch.int32)

    with torch.no_grad():
        image_embedding = model.forward_image(normalized)["backbone_fpn"][-1]
        image_embedding = image_embedding + model.no_mem_embed.permute(1, 2, 0).reshape(1, 256, 1, 1)
        sparse, dense = model.sam_prompt_encoder(
            points=(coordinates, labels),
            boxes=None,
            masks=prior,
        )
        masks, iou, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[],
        )
        resized = functional.interpolate(
            masks,
            size=(193, 257),
            mode="bilinear",
            align_corners=False,
        )

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mask_prompt_embedding_reference.bin": dense,
        "iterative_masks_reference.bin": masks,
        "iterative_iou_reference.bin": iou,
        "resized_masks_reference.bin": resized,
    }
    for filename, tensor in outputs.items():
        values = tensor.detach().float().contiguous().numpy().astype("<f4")
        (arguments.output_directory / filename).write_bytes(values.tobytes(order="C"))
        print(f"{filename}: shape={values.shape}, mean={values.mean():.9f}")


if __name__ == "__main__":
    main()
