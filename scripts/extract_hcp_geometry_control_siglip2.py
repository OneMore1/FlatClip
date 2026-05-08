from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_HCP_FLATMAP_ROOT = Path("outputs/flatmaps/hcp")
DEFAULT_HCP_ROI_ROOT = Path("data/roi/hcp/Schaefer2018_100")

from load_siglip2 import load_model


SUBJECT_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
ROI_RE = re.compile(r"(?P<subject>\d{6})__.*_(?P<start>\d{4})-(?P<stop>\d{4})\.npy$")


@dataclass(frozen=True)
class Task:
    subject_name: str
    output_file: Path
    frame_paths: tuple[Path, ...] = ()
    roi_file: Path | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract SigLIP2 NaFlex features for HCP geometry-control inputs.")
    p.add_argument(
        "--variant",
        choices=(
            "real_flatmap",
            "spatial_block_shuffle",
            "random_vertex_permutation",
            "random_image",
            "fc_heatmap",
            "left_right_swap",
            "within_hemi_region_shuffle",
            "phase_matched_randomization",
        ),
        required=True,
    )
    p.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_HCP_FLATMAP_ROOT,
    )
    p.add_argument(
        "--roi-root",
        type=Path,
        default=DEFAULT_HCP_ROI_ROOT,
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--siglip-variant", default="naflex")
    p.add_argument("--image-size", type=int, default=224, help="Fixed square canvas size. Use 0 to keep original flatmap size.")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument(
        "--region-block-size",
        type=int,
        default=16,
        help="Block size for image-level within-hemisphere region-like shuffle.",
    )
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--min-frames", type=int, default=40)
    p.add_argument("--fc-repeat-frames", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha1(("|".join(str(x) for x in (base_seed, *parts))).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def pad_to_patch_multiple(
    array: np.ndarray,
    mask: np.ndarray | None,
    patch_size: int,
    fill: int = 255,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = array.shape[:2]
    padded_h = math.ceil(h / patch_size) * patch_size
    padded_w = math.ceil(w / patch_size) * patch_size
    if (padded_h, padded_w) == (h, w):
        return array, mask
    out = np.full((padded_h, padded_w, 3), fill, dtype=np.uint8)
    out[:h, :w] = array
    mask_out = None
    if mask is not None:
        mask_out = np.zeros((padded_h, padded_w), dtype=bool)
        mask_out[:h, :w] = mask
    return out, mask_out


def rgba_to_rgb_and_mask(path: Path, patch_size: int, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    if image_size > 0:
        width, height = rgba.size
        scale = min(image_size / width, image_size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        rgba = rgba.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
        canvas = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
        left = (image_size - new_width) // 2
        top = (image_size - new_height) // 2
        canvas.alpha_composite(rgba, (left, top))
        rgba = canvas
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(rgba.getchannel("A"), dtype=np.uint8) > 0
    array, mask = pad_to_patch_multiple(array, mask, patch_size)
    assert mask is not None
    return array, mask


def template_shape(input_root: Path, patch_size: int, image_size: int) -> tuple[int, int]:
    first_subject = next(p for p in sorted(input_root.iterdir()) if p.is_dir())
    first_frame = next(iter(sorted(first_subject.glob("frame_*.png"))))
    array, _ = rgba_to_rgb_and_mask(first_frame, patch_size, image_size)
    return int(array.shape[0]), int(array.shape[1])


def build_block_permutation(height: int, width: int, patch_size: int, seed: int) -> np.ndarray:
    n_blocks = (height // patch_size) * (width // patch_size)
    rng = np.random.default_rng(seed)
    return rng.permutation(n_blocks)


def apply_block_shuffle(array: np.ndarray, permutation: np.ndarray, patch_size: int) -> np.ndarray:
    h, w, c = array.shape
    gh = h // patch_size
    gw = w // patch_size
    patches = (
        array.reshape(gh, patch_size, gw, patch_size, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(gh * gw, patch_size, patch_size, c)
    )
    shuffled = patches[permutation]
    return shuffled.reshape(gh, gw, patch_size, patch_size, c).transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def build_mask_permutation(input_root: Path, patch_size: int, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    first_subject = next(p for p in sorted(input_root.iterdir()) if p.is_dir())
    first_frame = next(iter(sorted(first_subject.glob("frame_*.png"))))
    _, mask = rgba_to_rgb_and_mask(first_frame, patch_size, image_size)
    positions = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(seed)
    return positions, rng.permutation(len(positions))


def apply_vertex_permutation(array: np.ndarray, positions: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    flat = array.reshape(-1, 3).copy()
    vals = flat[positions].copy()
    flat[positions] = vals[permutation]
    return flat.reshape(array.shape)


def apply_left_right_swap(array: np.ndarray) -> np.ndarray:
    mid = array.shape[1] // 2
    return np.concatenate([array[:, mid:], array[:, :mid]], axis=1)


def build_within_hemi_block_permutations(height: int, width: int, block_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mid = width // 2
    rng = np.random.default_rng(seed)
    left_n = (height // block_size) * (mid // block_size)
    right_n = (height // block_size) * ((width - mid) // block_size)
    return rng.permutation(left_n), rng.permutation(right_n)


def _shuffle_region_blocks(region: np.ndarray, permutation: np.ndarray, block_size: int) -> np.ndarray:
    h, w, c = region.shape
    h2 = (h // block_size) * block_size
    w2 = (w // block_size) * block_size
    if h2 == 0 or w2 == 0:
        return region.copy()
    out = region.copy()
    core = region[:h2, :w2]
    gh = h2 // block_size
    gw = w2 // block_size
    patches = (
        core.reshape(gh, block_size, gw, block_size, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(gh * gw, block_size, block_size, c)
    )
    shuffled = patches[permutation]
    out[:h2, :w2] = shuffled.reshape(gh, gw, block_size, block_size, c).transpose(0, 2, 1, 3, 4).reshape(h2, w2, c)
    return out


def apply_within_hemi_region_shuffle(
    array: np.ndarray,
    left_perm: np.ndarray,
    right_perm: np.ndarray,
    block_size: int,
) -> np.ndarray:
    mid = array.shape[1] // 2
    out = array.copy()
    out[:, :mid] = _shuffle_region_blocks(array[:, :mid], left_perm, block_size)
    out[:, mid:] = _shuffle_region_blocks(array[:, mid:], right_perm, block_size)
    return out


def _rank_match(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_template = np.sort(template)
    out = np.empty_like(template, dtype=np.float32)
    out[order] = sorted_template.astype(np.float32, copy=False)
    return out


def phase_matched_randomization(array: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    """Randomize phase while rank-matching the foreground histogram.

    The cortical foreground mask is kept fixed and the background is restored to
    white, so this control preserves the rendered flatmap silhouette and the
    foreground RGB histogram while disrupting spatial phase/geometry.
    """
    rng = np.random.default_rng(seed)
    out = np.full_like(array, 255, dtype=np.uint8)
    if int(mask.sum()) < 2:
        return out
    for ch in range(3):
        channel = array[:, :, ch].astype(np.float32)
        fg = channel[mask]
        fill = float(fg.mean())
        work = channel.copy()
        work[~mask] = fill
        centered = work - fill
        spectrum = np.fft.rfft2(centered)
        amplitude = np.abs(spectrum)
        phase = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
        phase[0, 0] = 0.0
        randomized = np.fft.irfft2(amplitude * np.exp(1j * phase), s=channel.shape).real
        matched = _rank_match(randomized[mask], fg)
        out[:, :, ch][mask] = np.clip(np.rint(matched), 0, 255).astype(np.uint8)
    return out


def random_image(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def parse_roi_file(path: Path) -> tuple[str, int]:
    match = ROI_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse HCP ROI file name: {path.name}")
    return match.group("subject"), int(match.group("start"))


def choose_one_roi_file_per_subject(roi_root: Path) -> list[tuple[str, Path]]:
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for split in ("train", "val", "test"):
        for path in (roi_root / split).glob("*.npy"):
            sid, start = parse_roi_file(path)
            grouped.setdefault(sid, []).append((start, path))
    return [(sid, sorted(items, key=lambda x: (x[0], x[1].name))[0][1]) for sid, items in sorted(grouped.items())]


def compute_fc(path: Path) -> np.ndarray:
    ts = np.load(path).astype(np.float32)
    if ts.ndim != 2:
        raise ValueError(f"Expected 2D ROI time series, got {path} shape={ts.shape}")
    if ts.shape[1] != 100 and ts.shape[0] == 100:
        ts = ts.T
    if ts.shape[1] != 100:
        raise ValueError(f"Expected 100 ROI columns, got {path} shape={ts.shape}")
    ts = ts[:200]
    ts = ts - ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    ts = ts / std
    fc = np.corrcoef(ts, rowvar=False).astype(np.float32)
    fc[~np.isfinite(fc)] = 0.0
    fc = np.clip(fc, -1.0, 1.0)
    np.fill_diagonal(fc, 1.0)
    return fc


def fc_to_image(fc: np.ndarray, height: int, width: int) -> np.ndarray:
    x = np.clip(fc, -1.0, 1.0)
    rgb = np.empty((*x.shape, 3), dtype=np.float32)
    pos = x >= 0
    rgb[pos, 0] = 255.0
    rgb[pos, 1] = 255.0 * (1.0 - x[pos])
    rgb[pos, 2] = 255.0 * (1.0 - x[pos])
    rgb[~pos, 0] = 255.0 * (1.0 + x[~pos])
    rgb[~pos, 1] = 255.0 * (1.0 + x[~pos])
    rgb[~pos, 2] = 255.0
    rgb = np.clip(rgb, 0.0, 255.0).round().astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


class ControlDataset(Dataset):
    def __init__(
        self,
        task: Task,
        args: argparse.Namespace,
        *,
        template_hw: tuple[int, int],
        block_perm: np.ndarray | None = None,
        mask_positions: np.ndarray | None = None,
        mask_perm: np.ndarray | None = None,
        hemi_perms: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.task = task
        self.args = args
        self.template_hw = template_hw
        self.block_perm = block_perm
        self.mask_positions = mask_positions
        self.mask_perm = mask_perm
        self.hemi_perms = hemi_perms
        self.fc_image: np.ndarray | None = None
        if args.variant == "fc_heatmap":
            self.names = [f"fc_repeat_{i:04d}.png" for i in range(args.fc_repeat_frames)]
            assert task.roi_file is not None
            self.fc_image = fc_to_image(compute_fc(task.roi_file), *template_hw)
        else:
            self.names = [path.name for path in task.frame_paths]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        args = self.args
        h, w = self.template_hw
        if args.variant == "fc_heatmap":
            assert self.fc_image is not None
            array = self.fc_image
        elif args.variant == "random_image":
            array = random_image(h, w, stable_seed(args.seed, self.task.subject_name, self.names[idx]))
        else:
            array, mask = rgba_to_rgb_and_mask(self.task.frame_paths[idx], args.patch_size, args.image_size)
            if args.variant == "spatial_block_shuffle":
                assert self.block_perm is not None
                array = apply_block_shuffle(array, self.block_perm, args.patch_size)
            elif args.variant == "random_vertex_permutation":
                assert self.mask_positions is not None and self.mask_perm is not None
                array = apply_vertex_permutation(array, self.mask_positions, self.mask_perm)
            elif args.variant == "left_right_swap":
                array = apply_left_right_swap(array)
            elif args.variant == "within_hemi_region_shuffle":
                assert self.hemi_perms is not None
                array = apply_within_hemi_region_shuffle(array, self.hemi_perms[0], self.hemi_perms[1], args.region_block_size)
            elif args.variant == "phase_matched_randomization":
                array = phase_matched_randomization(
                    array,
                    mask,
                    stable_seed(args.seed, "phase_matched_randomization", self.task.subject_name, self.names[idx]),
                )
        return Image.fromarray(array, mode="RGB"), self.names[idx]


def collate(batch: Sequence[tuple[Image.Image, str]]) -> tuple[list[Image.Image], list[str]]:
    return [item[0] for item in batch], [item[1] for item in batch]


def discover_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    out_root = args.output_root / "siglip2" / args.siglip_variant
    if args.variant == "fc_heatmap":
        for sid, roi_file in choose_one_roi_file_per_subject(args.roi_root):
            output_file = out_root / f"{sid}_FC_Schaefer100.npz"
            if output_file.exists() and not args.overwrite:
                continue
            tasks.append(Task(subject_name=f"{sid}_FC_Schaefer100", output_file=output_file, roi_file=roi_file))
    else:
        for source_dir in sorted(p for p in args.input_root.iterdir() if p.is_dir()):
            frame_paths = tuple(sorted(source_dir.glob("frame_*.png")))
            if len(frame_paths) < args.min_frames:
                continue
            frame_paths = frame_paths[: args.max_frames]
            output_file = out_root / f"{source_dir.name}.npz"
            if output_file.exists() and not args.overwrite:
                continue
            tasks.append(Task(subject_name=source_dir.name, output_file=output_file, frame_paths=frame_paths))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def batch_to_inputs(images: list[Image.Image], bundle, device: str) -> dict[str, object]:
    patch_size = int(getattr(bundle.image_processor, "patch_size", 16))
    max_num_patches = max((im.size[0] // patch_size) * (im.size[1] // patch_size) for im in images)
    inputs = dict(
        bundle.image_processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            max_num_patches=max_num_patches,
        )
    )
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in inputs.items()}


def call_with_supported_kwargs(fn: object, kwargs: dict[str, object]) -> object:
    target = fn.forward if isinstance(fn, torch.nn.Module) else fn
    signature = inspect.signature(target)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        supported = kwargs
    else:
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**supported)


def cast_array(array: np.ndarray, save_dtype: str) -> np.ndarray:
    return array.astype(np.float16 if save_dtype == "float16" else np.float32, copy=False)


def extract_one(task: Task, bundle, args: argparse.Namespace, template_hw: tuple[int, int], block_perm, mask_positions, mask_perm, hemi_perms) -> None:
    dataset = ControlDataset(
        task,
        args,
        template_hw=template_hw,
        block_perm=block_perm,
        mask_positions=mask_positions,
        mask_perm=mask_perm,
        hemi_perms=hemi_perms,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )
    names: list[str] = []
    cls_chunks: list[np.ndarray] = []
    for images, batch_names in loader:
        inputs = batch_to_inputs(images, bundle, args.device)
        with torch.inference_mode():
            image_embeds = call_with_supported_kwargs(bundle.model.get_image_features, inputs)
        cls_chunks.append(cast_array(image_embeds.detach().cpu().numpy(), args.save_dtype))
        names.extend(batch_names)
    cls_all = np.concatenate(cls_chunks, axis=0)
    out = {
        "cls": cls_all,
        "image_embeds": cls_all,
        "frame_names": np.asarray(names),
    }

    task.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(task.output_file, **out)
    metadata = {
        "variant": args.variant,
        "subject_name": task.subject_name,
        "roi_file": str(task.roi_file) if task.roi_file else None,
        "siglip_variant": args.siglip_variant,
        "checkpoint": str(getattr(bundle, "checkpoint_dir", "")),
        "frame_count": int(cls_all.shape[0]),
        "cls_shape": list(cls_all.shape),
        "save_dtype": args.save_dtype,
        "control_seed": args.seed,
        "preprocess": (
            "fixed_square_canvas_then_siglip2_naflex_do_resize_false"
            if args.image_size > 0
            else "original_flatmap_size_padded_to_patch_multiple_then_siglip2_naflex_do_resize_false"
        ),
        "image_size": int(args.image_size),
        "region_block_size": int(args.region_block_size),
    }
    if args.variant == "left_right_swap":
        metadata["control_description"] = "image-level swap of left and right halves of the fixed flatmap canvas"
    elif args.variant == "within_hemi_region_shuffle":
        metadata["control_description"] = (
            "image-level region-like shuffle: 16x16 blocks are permuted independently within the left and right canvas halves"
        )
    elif args.variant == "phase_matched_randomization":
        metadata["control_description"] = (
            "phase-randomized foreground with per-channel foreground histogram rank matching; background restored to white"
        )
    task.output_file.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    tasks = discover_tasks(args)
    assigned = [task for idx, task in enumerate(tasks) if idx % args.world_size == args.rank]
    print(
        f"variant={args.variant} rank={args.rank}/{args.world_size} tasks_total={len(tasks)} "
        f"assigned={len(assigned)} output={args.output_root}",
        flush=True,
    )
    if not assigned:
        return 0

    template_hw = template_shape(args.input_root, args.patch_size, args.image_size)
    print(f"template_hw={template_hw}", flush=True)
    block_perm = None
    mask_positions = None
    mask_perm = None
    hemi_perms = None
    if args.variant == "spatial_block_shuffle":
        block_perm = build_block_permutation(*template_hw, args.patch_size, args.seed)
    elif args.variant == "random_vertex_permutation":
        mask_positions, mask_perm = build_mask_permutation(args.input_root, args.patch_size, args.image_size, args.seed)
        print(f"mask_pixels={len(mask_positions)}", flush=True)
    elif args.variant == "within_hemi_region_shuffle":
        hemi_perms = build_within_hemi_block_permutations(*template_hw, args.region_block_size, args.seed)
        print(f"within_hemi_blocks={len(hemi_perms[0])}+{len(hemi_perms[1])}", flush=True)

    bundle = load_model(args.device, variant=args.siglip_variant)
    for idx, task in enumerate(assigned, start=1):
        print(f"[{idx}/{len(assigned)}] {task.subject_name}", flush=True)
        extract_one(task, bundle, args, template_hw, block_perm, mask_positions, mask_perm, hemi_perms)
    print(f"saved {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
