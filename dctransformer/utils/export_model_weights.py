#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from dctransformer.utils.runtime_utils import extract_model_state_dict


def torch_load_any(path: str | Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_suffix(".pt")
    return input_path.with_name(f"{input_path.name}.pt")


def count_params(state_dict: dict[str, Any]) -> int:
    return sum(int(value.numel()) for value in state_dict.values() if torch.is_tensor(value))


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a checkpoint to a publishable .pt file that contains model weights only."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input checkpoint/state_dict file (.ckpt or .pt).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Path to output weights file. Defaults to the input path with a .pt suffix.",
    )
    parser.add_argument(
        "--keep-module-prefix",
        action="store_true",
        help="Keep 'module.' prefixes instead of stripping them from DataParallel checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path).resolve()
    )
    if output_path == input_path:
        raise ValueError("Refusing to overwrite the input checkpoint. Please choose a different --output path.")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --force to overwrite it.")

    raw_obj = torch_load_any(input_path, map_location="cpu")
    state_dict = extract_model_state_dict(
        raw_obj,
        strip_module_prefix=not bool(args.keep_module_prefix),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)

    tensor_count = len(state_dict)
    param_count = count_params(state_dict)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Tensors: {tensor_count}")
    print(f"Params:  {param_count}")
    print(f"Stripped module prefix: {not bool(args.keep_module_prefix)}")


if __name__ == "__main__":
    main()
