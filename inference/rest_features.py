from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_ROOT = Path("<path>")
DEFAULT_OUTPUT_ROOT = Path("<path>")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LOADER_MODULES = {
    "siglip2": "inference.siglip2",
}


@dataclass(frozen=True)
class Task:
    source_dir: Path
    output_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract flatmap features with a frozen SigLIP2 encoder.")
    parser.add_argument("--model-family", choices=tuple(LOADER_MODULES), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit-dirs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--bg-color", default="255,255,255")
    parser.add_argument("--save-full-highdim", action="store_true")
    parser.add_argument(
        "--no-resize-original",
        action="store_true",
        help="For NaFlex-style processors, keep original spatial size and only pad to a patch-size multiple.",
    )
    return parser.parse_args()


def parse_bg_color(text: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3 or any(x < 0 or x > 255 for x in parts):
        raise ValueError(f"Invalid bg-color: {text}")
    return tuple(parts)


def discover_png_dirs(input_root: Path, output_root: Path) -> list[Path]:
    png_dirs: list[Path] = []
    output_root = output_root.resolve()
    for current_root, dirnames, filenames in os.walk(input_root):
        current_path = Path(current_root).resolve()
        if current_path == output_root or output_root in current_path.parents:
            dirnames[:] = []
            continue
        if any(name.lower().endswith(".png") for name in filenames):
            png_dirs.append(Path(current_root))
    return sorted(png_dirs)


def build_tasks(args: argparse.Namespace) -> list[Task]:
    model_output_root = args.output_root / args.model_family / args.variant
    tasks: list[Task] = []
    for source_dir in discover_png_dirs(args.input_root, args.output_root):
        output_file = model_output_root / f"{source_dir.name}.npz"
        if not args.overwrite and output_file.exists():
            continue
        tasks.append(Task(source_dir=source_dir, output_file=output_file))
    if args.limit_dirs is not None:
        tasks = tasks[: args.limit_dirs]
    return tasks


def iter_assigned_tasks(tasks: Sequence[Task], rank: int, world_size: int) -> list[Task]:
    return [task for index, task in enumerate(tasks) if index % world_size == rank]


class FrameDataset(Dataset):
    def __init__(self, frame_paths: Sequence[Path], bg_color: tuple[int, int, int]):
        self.frame_paths = list(frame_paths)
        self.bg_color = bg_color

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int):
        path = self.frame_paths[index]
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (*self.bg_color, 255))
            rgb = Image.alpha_composite(background, rgba).convert("RGB")
        return rgb, path.name


def collate_batch(batch):
    images = [item[0] for item in batch]
    names = [item[1] for item in batch]
    return images, names


def preprocess_image(
    image: Image.Image,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = image.resize((new_width, new_height), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2
    canvas.paste(resized, (left, top))
    tensor = TF.to_tensor(canvas)
    return TF.normalize(tensor, mean, std)


def load_feature_api(model_family: str) -> tuple[Callable, Callable]:
    module = importlib.import_module(LOADER_MODULES[model_family])
    if not hasattr(module, "load_model") or not hasattr(module, "extract_features"):
        raise AttributeError(f"Loader {module.__name__} must expose load_model() and extract_features()")
    return module.load_model, module.extract_features


def load_bundle(args: argparse.Namespace):
    load_model, extract_features = load_feature_api(args.model_family)
    load_model_sig = inspect.signature(load_model)
    if "variant" in load_model_sig.parameters:
        model_or_bundle = load_model(args.device, variant=args.variant)
    else:
        model_or_bundle = load_model(args.device)
    return model_or_bundle, extract_features


def maybe_adjust_supported_resolution(pixel_values: torch.Tensor, model_or_bundle) -> torch.Tensor:
    model = getattr(model_or_bundle, "model", None)
    if model is None or not hasattr(model, "get_nearest_supported_resolution"):
        return pixel_values
    height, width = pixel_values.shape[-2:]
    nearest = model.get_nearest_supported_resolution(height, width)
    target_height = int(nearest.height)
    target_width = int(nearest.width)
    if (target_height, target_width) == (height, width):
        return pixel_values
    return F.interpolate(
        pixel_values,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def pad_images_to_patch_multiple(
    images: list[Image.Image],
    patch_size: int,
    bg_color: tuple[int, int, int],
) -> tuple[list[Image.Image], int]:
    padded_images: list[Image.Image] = []
    max_num_patches = 0
    for image in images:
        width, height = image.size
        padded_width = math.ceil(width / patch_size) * patch_size
        padded_height = math.ceil(height / patch_size) * patch_size
        padded = Image.new("RGB", (padded_width, padded_height), bg_color)
        padded.paste(image, (0, 0))
        padded_images.append(padded)
        max_num_patches = max(
            max_num_patches,
            (padded_width // patch_size) * (padded_height // patch_size),
        )
    return padded_images, max_num_patches


def batch_to_model_inputs(
    images: list[Image.Image],
    model_or_bundle,
    fallback_image_size: int,
    *,
    no_resize_original: bool,
    bg_color: tuple[int, int, int],
) -> dict[str, object]:
    if hasattr(model_or_bundle, "image_processor"):
        if no_resize_original:
            patch_size = int(getattr(model_or_bundle.image_processor, "patch_size", 16))
            images, max_num_patches = pad_images_to_patch_multiple(images, patch_size, bg_color)
            inputs = dict(
                model_or_bundle.image_processor(
                    images=images,
                    return_tensors="pt",
                    do_resize=False,
                    max_num_patches=max_num_patches,
                )
            )
        else:
            inputs = dict(model_or_bundle.image_processor(images=images, return_tensors="pt"))
        if set(inputs) == {"pixel_values"}:
            inputs["pixel_values"] = maybe_adjust_supported_resolution(inputs["pixel_values"], model_or_bundle)
        return inputs
    if hasattr(model_or_bundle, "processor"):
        inputs = dict(model_or_bundle.processor(images=images, return_tensors="pt", do_resize=True))
        if "pixel_values" in inputs:
            inputs["pixel_values"] = maybe_adjust_supported_resolution(inputs["pixel_values"], model_or_bundle)
        return inputs

    image_size = getattr(model_or_bundle, "image_size", fallback_image_size)
    mean = getattr(model_or_bundle, "image_mean", IMAGENET_MEAN)
    std = getattr(model_or_bundle, "image_std", IMAGENET_STD)
    tensors = [preprocess_image(image, image_size, mean, std) for image in images]
    pixel_values = torch.stack(tensors, dim=0)
    return {"pixel_values": maybe_adjust_supported_resolution(pixel_values, model_or_bundle)}


def move_model_inputs_to_device(inputs: dict[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in inputs.items()
    }


def cast_feature(array: np.ndarray, save_dtype: str) -> np.ndarray:
    return array.astype(np.float16 if save_dtype == "float16" else np.float32, copy=False)


def compact_feature_map(
    feature_name: str, tensor: torch.Tensor, save_dtype: str, save_full_highdim: bool
) -> dict[str, np.ndarray]:
    array = tensor.detach().cpu().numpy()
    if array.ndim <= 2:
        return {feature_name: cast_feature(array, save_dtype)}

    outputs: dict[str, np.ndarray] = {}
    if save_full_highdim:
        outputs[feature_name] = cast_feature(array, save_dtype)
    if array.ndim == 3:
        outputs[f"{feature_name}_mean"] = cast_feature(array.mean(axis=1), save_dtype)
    elif array.ndim == 4:
        outputs[f"{feature_name}_mean"] = cast_feature(array.mean(axis=(-2, -1)), save_dtype)
    else:
        reduce_axes = tuple(range(1, array.ndim))
        outputs[f"{feature_name}_mean"] = cast_feature(array.mean(axis=reduce_axes), save_dtype)
    return outputs


def extract_directory_features(
    task: Task,
    model_or_bundle,
    extract_features: Callable,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    save_dtype: str,
    bg_color: tuple[int, int, int],
    model_family: str,
    variant: str,
    save_full_highdim: bool,
    no_resize_original: bool,
) -> None:
    frame_paths = sorted(task.source_dir.glob("*.png"))
    if not frame_paths:
        return

    dataset = FrameDataset(frame_paths, bg_color=bg_color)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_batch,
    )

    collected: dict[str, list[np.ndarray]] = {}
    all_names: list[str] = []
    for images, frame_names in loader:
        model_inputs = move_model_inputs_to_device(
            batch_to_model_inputs(
                images,
                model_or_bundle,
                image_size,
                no_resize_original=no_resize_original,
                bg_color=bg_color,
            ),
            device,
        )
        with torch.inference_mode():
            if set(model_inputs) == {"pixel_values"}:
                feature_dict = extract_features(model_or_bundle, model_inputs["pixel_values"])
            else:
                feature_dict = extract_features(model_or_bundle, **model_inputs)
        for key, value in feature_dict.items():
            compacted = compact_feature_map(key, value, save_dtype, save_full_highdim)
            for compact_key, compact_value in compacted.items():
                collected.setdefault(compact_key, []).append(compact_value)
        all_names.extend(frame_names)

    payload = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    payload["frame_names"] = np.asarray(all_names)
    payload["metadata"] = np.asarray(
        json.dumps(
            {
                "source_name": task.source_dir.name,
                "frame_count": len(frame_paths),
                "model_family": model_family,
                "variant": variant,
                "image_size": getattr(model_or_bundle, "image_size", image_size),
                "save_dtype": save_dtype,
                "save_full_highdim": save_full_highdim,
                "no_resize_original": no_resize_original,
                "feature_keys": sorted(payload.keys()),
            }
        )
    )

    task.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(task.output_file, **payload)


def main() -> int:
    args = parse_args()
    bg_color = parse_bg_color(args.bg_color)
    tasks = build_tasks(args)
    assigned_tasks = iter_assigned_tasks(tasks, rank=args.rank, world_size=args.world_size)
    print(
        f"model_family={args.model_family} variant={args.variant} "
        f"tasks_total={len(tasks)} tasks_assigned={len(assigned_tasks)} "
        f"rank={args.rank} world_size={args.world_size} device={args.device}",
        flush=True,
    )
    if not assigned_tasks:
        return 0

    model_or_bundle, extract_features = load_bundle(args)
    for index, task in enumerate(assigned_tasks, start=1):
        print(
            f"[{index}/{len(assigned_tasks)}] {task.source_dir} -> {task.output_file}",
            flush=True,
        )
        extract_directory_features(
            task,
            model_or_bundle,
            extract_features,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            save_dtype=args.save_dtype,
            bg_color=bg_color,
            model_family=args.model_family,
            variant=args.variant,
            save_full_highdim=args.save_full_highdim,
            no_resize_original=args.no_resize_original,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
