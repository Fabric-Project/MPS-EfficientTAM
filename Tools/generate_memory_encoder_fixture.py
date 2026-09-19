#!/usr/bin/env python3
"""Generate official PyTorch spatial-memory fixtures."""

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

    with torch.no_grad():
        raw_image_embedding = model.forward_image(normalized)["backbone_fpn"][-1]
        decoder_embedding = raw_image_embedding + model.no_mem_embed.permute(1, 2, 0).reshape(1, 256, 1, 1)
        coordinates = torch.tensor([[[160.0, 240.0]]], dtype=torch.float32)
        labels = torch.tensor([[1]], dtype=torch.int32)
        sparse, dense = model.sam_prompt_encoder(points=(coordinates, labels), boxes=None, masks=None)
        masks, iou, _, _ = model.sam_mask_decoder(
            image_embeddings=decoder_embedding,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[],
        )
        selected_index = iou.argmax(dim=1)
        selected = masks[torch.arange(masks.shape[0]), selected_index].unsqueeze(1)
        high_resolution = functional.interpolate(
            selected,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
        memory_mask = torch.sigmoid(high_resolution) * 20.0 - 10.0
        memory = model.memory_encoder(raw_image_embedding, memory_mask, skip_mask_sigmoid=True)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "memory_selected_mask_reference.bin": selected,
        "memory_features_reference.bin": memory["vision_features"],
        "memory_position_reference.bin": memory["vision_pos_enc"][0],
    }
    for filename, tensor in outputs.items():
        values = tensor.detach().float().contiguous().numpy().astype("<f4")
        (arguments.output_directory / filename).write_bytes(values.tobytes(order="C"))
        print(f"{filename}: shape={values.shape}, mean={values.mean():.9f}")


if __name__ == "__main__":
    main()
