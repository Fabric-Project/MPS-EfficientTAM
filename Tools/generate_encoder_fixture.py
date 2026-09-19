#!/usr/bin/env python3
"""Generate the full official-PyTorch image-encoder accuracy fixture."""

import argparse
import pathlib
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream_checkout", type=pathlib.Path)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.upstream_checkout))
    # Upstream deliberately uses bilinear absolute-position interpolation on
    # Apple MPS because bicubic is unsupported there. Match that production
    # path even when this fixture script itself is run on CPU.
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
        result = model.forward_image(normalized)["backbone_fpn"][-1]
        result = result + model.no_mem_embed.permute(1, 2, 0).reshape(1, 256, 1, 1)

    output = result.detach().float().contiguous().numpy().astype("<f4")
    arguments.output.write_bytes(output.tobytes(order="C"))
    print(
        f"Wrote {output.size} values; mean={output.mean():.9f}; "
        f"min={output.min():.9f}; max={output.max():.9f}"
    )


if __name__ == "__main__":
    main()
