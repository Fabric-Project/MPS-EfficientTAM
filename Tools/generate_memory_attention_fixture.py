#!/usr/bin/env python3
"""Generate a one-memory, one-object EfficientTAM attention fixture."""

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as functional


def normalized_image(offset: int) -> torch.Tensor:
    count = 512 * 512 * 3
    rgb = (torch.arange(count, dtype=torch.int64) + offset).remainder(251).float() / 250.0
    rgb = rgb.reshape(1, 512, 512, 3)
    rgb = (rgb - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    return rgb.permute(0, 3, 1, 2)


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

    with torch.no_grad():
        first = model.forward_image(normalized_image(0))
        first_raw = first["backbone_fpn"][-1]
        first_decoder = first_raw + model.no_mem_embed.permute(1, 2, 0).reshape(1, 256, 1, 1)
        coordinates = torch.tensor([[[160.0, 240.0]]], dtype=torch.float32)
        labels = torch.tensor([[1]], dtype=torch.int32)
        sparse, dense = model.sam_prompt_encoder(points=(coordinates, labels), boxes=None, masks=None)
        masks, iou, tokens, object_score = model.sam_mask_decoder(
            image_embeddings=first_decoder,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[],
        )
        best = iou.argmax(dim=1)
        selected = masks[torch.arange(1), best].unsqueeze(1)
        selected_token = tokens[torch.arange(1), best]
        pointer = model.obj_ptr_proj(selected_token)
        present = (object_score > 0).float()
        pointer = present * pointer + (1 - present) * model.no_obj_ptr
        high_resolution = functional.interpolate(selected, (512, 512), mode="bilinear", align_corners=False)
        memory_mask = torch.sigmoid(high_resolution) * 20.0 - 10.0
        encoded = model.memory_encoder(first_raw, memory_mask, skip_mask_sigmoid=True)
        memory_features = encoded["vision_features"]
        memory_position = encoded["vision_pos_enc"][0] + model.maskmem_tpos_enc[0].reshape(1, 64, 1, 1)

        second = model.forward_image(normalized_image(37))
        second_raw = second["backbone_fpn"][-1]
        second_decoder = second_raw + model.no_mem_embed.permute(1, 2, 0).reshape(1, 256, 1, 1)
        current = second_raw.flatten(2).permute(2, 0, 1)
        current_position = second["vision_pos_enc"][-1].flatten(2).permute(2, 0, 1)
        spatial_memory = memory_features.flatten(2).permute(2, 0, 1)
        spatial_position = memory_position.flatten(2).permute(2, 0, 1)
        pointer_tokens = pointer.reshape(1, 4, 64).permute(1, 0, 2)
        pointer_position = torch.zeros_like(pointer_tokens)
        attended = model.memory_attention(
            curr=[current],
            curr_pos=[current_position],
            memory=torch.cat([spatial_memory, pointer_tokens], dim=0),
            memory_pos=torch.cat([spatial_position, pointer_position], dim=0),
            num_obj_ptr_tokens=4,
        )
        attended = attended.permute(1, 2, 0).reshape(1, 256, 32, 32)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "attention_current_embedding_reference.bin": second_decoder,
        "attention_memory_features_reference.bin": memory_features,
        "attention_memory_position_reference.bin": memory_position,
        "attention_object_pointer_reference.bin": pointer,
        "attention_output_reference.bin": attended,
    }
    for filename, tensor in outputs.items():
        values = tensor.detach().float().contiguous().numpy().astype("<f4")
        (arguments.output_directory / filename).write_bytes(values.tobytes(order="C"))
        print(f"{filename}: shape={values.shape}, mean={values.mean():.9f}")


if __name__ == "__main__":
    main()
