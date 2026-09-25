#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


def parse_nsd_id(path: Path) -> int:
    match = re.search(r"_(\d+)\.png$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse NSD id from {path}")
    return int(match.group(1))


def rgba_to_rgb(path: Path, bg_color: tuple[int, int, int]) -> Image.Image:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*bg_color, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


class FlatmapDataset(Dataset[tuple[Image.Image, int, str]]):
    def __init__(self, paths: list[Path], bg_color: tuple[int, int, int]):
        self.paths = paths
        self.bg_color = bg_color

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        path = self.paths[index]
        return rgba_to_rgb(path, self.bg_color), parse_nsd_id(path), str(path)


def collate(batch):
    images, nsd_ids, paths = zip(*batch, strict=True)
    return list(images), np.asarray(nsd_ids, dtype=np.int32), list(paths)


def parse_bg_color(text: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3 or any(x < 0 or x > 255 for x in parts):
        raise ValueError(f"Invalid bg-color: {text}")
    return tuple(parts)


def move_to_device(inputs: dict[str, object], device: str | torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in inputs.items()
    }


def call_with_supported_kwargs(fn: object, kwargs: dict[str, object]) -> object:
    import inspect

    target = fn.forward if isinstance(fn, torch.nn.Module) else fn
    signature = inspect.signature(target)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        supported = kwargs
    else:
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**supported)


def pool_patches(patch_tokens: torch.Tensor, spatial: torch.Tensor | None, output_patch_grid: int) -> torch.Tensor:
    if output_patch_grid <= 0:
        return patch_tokens
    pooled_rows = []
    for index in range(patch_tokens.shape[0]):
        patches = patch_tokens[index]
        if spatial is not None:
            h, w = [int(x) for x in spatial[index].detach().cpu().tolist()]
        else:
            h = w = round(patches.shape[0] ** 0.5)
        n_valid = h * w
        if n_valid > int(patches.shape[0]):
            raise ValueError(f"Patch count {patches.shape[0]} is smaller than spatial shape {(h, w)}")
        patches = patches[:n_valid]
        patch_tensor = patches.T.reshape(1, patches.shape[1], h, w)
        pooled = F.adaptive_avg_pool2d(patch_tensor.float(), (output_patch_grid, output_patch_grid))
        pooled_rows.append(pooled.reshape(patches.shape[1], output_patch_grid * output_patch_grid).T)
    return torch.stack(pooled_rows, dim=0)


def extract_batch(
    *,
    model: torch.nn.Module,
    processor: object,
    images: list[Image.Image],
    device: str,
    max_num_patches: int,
    output_patch_grid: int,
) -> torch.Tensor:
    inputs = dict(processor(images=images, return_tensors="pt", max_num_patches=max_num_patches))
    inputs = move_to_device(inputs, device)
    vision_inputs = {**inputs, "output_hidden_states": False, "return_dict": True}
    if "attention_mask" not in vision_inputs and "pixel_attention_mask" in vision_inputs:
        vision_inputs["attention_mask"] = vision_inputs["pixel_attention_mask"]
    outputs = call_with_supported_kwargs(model.vision_model, vision_inputs)
    patch_tokens = outputs.last_hidden_state
    spatial = inputs.get("spatial_shapes")
    return pool_patches(patch_tokens, spatial if torch.is_tensor(spatial) else None, output_patch_grid)


def save_manifest(output_file: Path, payload: dict[str, object]) -> None:
    manifest = output_file.with_suffix(".json")
    import json

    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("FLATCLIP_CHECKPOINT_ROOT", "<path>")),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-num-patches", type=int, default=1024)
    parser.add_argument("--output-patch-grid", type=int, default=16)
    parser.add_argument("--bg-color", default="255,255,255")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.input_root.glob("*.png"), key=parse_nsd_id)
    if not paths:
        raise FileNotFoundError(f"No PNG files found under {args.input_root}")
    if args.output_file.exists() and not args.overwrite:
        print(f"[skip] exists: {args.output_file}", flush=True)
        return
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} input={args.input_root} n={len(paths)} output={args.output_file}",
        flush=True,
    )
    processor = AutoImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(args.checkpoint, local_files_only=True).eval().to(device)

    dataset = FlatmapDataset(paths, parse_bg_color(args.bg_color))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate,
    )

    with torch.inference_mode():
        first_images, _, _ = next(iter(loader))
        probe = extract_batch(
            model=model,
            processor=processor,
            images=first_images[:1],
            device=device,
            max_num_patches=args.max_num_patches,
            output_patch_grid=args.output_patch_grid,
        )
    token_shape = tuple(int(x) for x in probe.shape[1:])
    nsd_ids = np.asarray([parse_nsd_id(path) for path in paths], dtype=np.int32)
    source_paths = np.asarray([path.name for path in paths], dtype=h5py.string_dtype(encoding="utf-8"))

    tmp_file = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    with h5py.File(tmp_file, "w") as h5:
        emb = h5.create_dataset("embeddings", shape=(len(paths), *token_shape), dtype="float32")
        h5.create_dataset("nsd_ids", data=nsd_ids)
        h5.create_dataset("source_png", data=source_paths)
        h5.attrs["backbone"] = "siglip2_base_patch16_naflex"
        h5.attrs["checkpoint"] = args.checkpoint.name
        h5.attrs["output_patch_grid"] = int(args.output_patch_grid)
        h5.attrs["max_num_patches"] = int(args.max_num_patches)

        offset = 0
        with torch.inference_mode():
            for batch_index, (images, batch_nsd_ids, _) in enumerate(loader, start=1):
                batch = extract_batch(
                    model=model,
                    processor=processor,
                    images=images,
                    device=device,
                    max_num_patches=args.max_num_patches,
                    output_patch_grid=args.output_patch_grid,
                )
                arr = batch.detach().cpu().float().numpy()
                emb[offset : offset + arr.shape[0]] = arr
                offset += arr.shape[0]
                if batch_index <= 3 or batch_index % 10 == 0:
                    print(
                        f"[{offset}/{len(paths)}] last_nsd={int(batch_nsd_ids[-1])}",
                        flush=True,
                    )

    tmp_file.replace(args.output_file)
    save_manifest(
        args.output_file,
        {
            "input_name": args.input_root.name,
            "output_name": args.output_file.name,
            "checkpoint": args.checkpoint.name,
            "n_images": len(paths),
            "embedding_shape": [len(paths), *token_shape],
            "dtype": "float32",
        },
    )
    print(f"[done] {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
