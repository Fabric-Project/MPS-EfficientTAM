#!/usr/bin/env python3
"""Export the official EfficientTAM Tiny 512 model to mapped float32 weights."""

import argparse
import json
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output_binary", type=pathlib.Path)
    parser.add_argument("output_manifest", type=pathlib.Path)
    arguments = parser.parse_args()

    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)["model"]
    selected = {
        name: tensor.detach().float().contiguous()
        for name, tensor in state.items()
    }

    manifest = {}
    offset = 0
    with arguments.output_binary.open("wb") as binary:
        for name in sorted(selected):
            tensor = selected[name]
            raw = tensor.numpy().tobytes(order="C")
            binary.write(raw)
            manifest[name] = {
                "offset": offset,
                "count": tensor.numel(),
                "shape": list(tensor.shape),
                "dtype": "float32",
            }
            offset += tensor.numel()

    arguments.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(selected)} tensors and {offset * 4} bytes")


if __name__ == "__main__":
    main()
