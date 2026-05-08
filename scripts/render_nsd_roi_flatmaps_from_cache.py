#!/usr/bin/env python
"""Render NSD fsLR32k average surfaces into ROI-only flatmap PNGs.

This renderer uses the existing PyCortex flatmask/flatverts cache directly, so
it does not need pycortex or matplotlib at runtime. It is intended for remote
batch rendering before frozen SigLIP2 feature extraction.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse


RDBU_R = np.asarray(
    [
        [5, 48, 97],
        [33, 102, 172],
        [67, 147, 195],
        [146, 197, 222],
        [209, 229, 240],
        [247, 247, 247],
        [253, 219, 199],
        [244, 165, 130],
        [214, 96, 77],
        [178, 24, 43],
        [103, 0, 31],
    ],
    dtype=np.float32,
)


def discover_npz_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("sub*_*.npz"))


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        left = np.asarray(data["left"]).reshape(-1).astype(bool)
        right = np.asarray(data["right"]).reshape(-1).astype(bool)
    if left.shape != (32492,) or right.shape != (32492,):
        raise ValueError(f"{path} yielded {left.shape}, {right.shape}; expected (32492,) each")
    return left, right


def load_flat_cache(cache_dir: Path):
    mask_npz = np.load(cache_dir / "flatmask_384.npz")
    flatmask = np.asarray(mask_npz["mask"], dtype=bool)
    flat_npz = np.load(cache_dir / "flatverts_384.npz")
    pixmap = sparse.csr_matrix(
        (
            flat_npz["data"],
            flat_npz["indices"],
            flat_npz["indptr"],
        ),
        shape=tuple(int(x) for x in flat_npz["shape"]),
    )
    if int(flatmask.sum()) != int(pixmap.shape[0]):
        raise ValueError(f"flatmask true count {flatmask.sum()} != pixmap rows {pixmap.shape[0]}")
    return flatmask, pixmap


def load_vector(path: Path, roi_mask: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, dict[str, object]]:
    left_mask, right_mask = roi_mask
    with np.load(path, allow_pickle=True) as data:
        left = np.asarray(data["lh"], dtype=np.float32).reshape(-1)
        right = np.asarray(data["rh"], dtype=np.float32).reshape(-1)
        meta = {
            key: data[key].tolist() if hasattr(data[key], "tolist") else data[key]
            for key in data.files
            if key not in ("lh", "rh")
        }
    if left.shape != (32492,) or right.shape != (32492,):
        raise ValueError(f"{path} yielded lh={left.shape}, rh={right.shape}; expected (32492,)")
    left = left.copy()
    right = right.copy()
    left[~left_mask] = np.nan
    right[~right_mask] = np.nan
    return np.concatenate([left, right], axis=0).astype(np.float32, copy=False), meta


def zscore_clip(vec: np.ndarray, clip: float) -> np.ndarray:
    out = vec.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        return out
    mu = float(out[finite].mean())
    sd = float(out[finite].std())
    if sd <= 1e-8:
        out[finite] = 0.0
    else:
        out[finite] = (out[finite] - mu) / sd
    out[finite] = np.clip(out[finite], -clip, clip)
    return out


def render_values(vec: np.ndarray, flatmask: np.ndarray, pixmap) -> np.ndarray:
    valid = np.isfinite(vec)
    values = pixmap.dot(np.nan_to_num(vec, nan=0.0))
    weights = pixmap.dot(valid.astype(np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        mapped = values / weights
    mapped = np.asarray(mapped, dtype=np.float32).reshape(-1)
    mapped[weights <= 0] = np.nan

    img = np.full(flatmask.shape, np.nan, dtype=np.float32)
    img[flatmask] = mapped
    img = img.T[::-1]
    return img.copy(order="C")


def colorize_rdBu_r(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    finite = np.isfinite(img)
    denom = max(vmax - vmin, 1e-6)
    pos = np.clip((np.nan_to_num(img, nan=vmin) - vmin) / denom, 0.0, 1.0)
    scaled = pos * (len(RDBU_R) - 1)
    lo = np.floor(scaled).astype(np.int16)
    hi = np.clip(lo + 1, 0, len(RDBU_R) - 1)
    frac = (scaled - lo)[..., None]
    rgb = RDBU_R[lo] * (1.0 - frac) + RDBU_R[hi] * frac
    rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    rgba[..., 3] = finite.astype(np.uint8) * 255
    return rgba


def crop_rgba(rgba: np.ndarray, pad: int) -> np.ndarray:
    ys, xs = np.where(rgba[..., 3] > 0)
    if ys.size == 0:
        return rgba
    top = max(0, int(ys.min()) - pad)
    bottom = min(rgba.shape[0], int(ys.max()) + 1 + pad)
    left = max(0, int(xs.min()) - pad)
    right = min(rgba.shape[1], int(xs.max()) + 1 + pad)
    return rgba[top:bottom, left:right]


def parse_subject_id(path: Path) -> int | None:
    match = re.match(r"sub(\d+)_", path.stem)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roi-mask", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("filestore/NSD_fsLR32k/cache"),
    )
    parser.add_argument("--zscore-clip", type=float, default=3.0)
    parser.add_argument("--crop-pad", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = discover_npz_files(args.input_root)
    assigned = [path for index, path in enumerate(files) if index % args.world_size == args.rank]
    if not files:
        raise FileNotFoundError(f"No input .npz files under {args.input_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roi_mask = load_mask(args.roi_mask)
    flatmask, pixmap = load_flat_cache(args.cache_dir)
    print(
        f"rank={args.rank} world_size={args.world_size} assigned={len(assigned)} "
        f"input_root={args.input_root} out_dir={args.out_dir}",
        flush=True,
    )

    for i, path in enumerate(assigned, start=1):
        out_png = args.out_dir / f"{path.stem}.png"
        out_json = args.out_dir / f"{path.stem}.json"
        if out_png.exists() and out_json.exists() and not args.overwrite:
            if i <= 3 or i % 500 == 0:
                print(f"[{i}/{len(assigned)}] skip {path.name}", flush=True)
            continue
        vec, meta = load_vector(path, roi_mask)
        vec = zscore_clip(vec, args.zscore_clip)
        img = render_values(vec, flatmask, pixmap)
        rgba = crop_rgba(colorize_rdBu_r(img, -args.zscore_clip, args.zscore_clip), args.crop_pad)
        Image.fromarray(rgba, mode="RGBA").save(out_png)
        meta.update(
            {
                "source_npz": str(path),
                "roi_mask": str(args.roi_mask),
                "norm_mode": "zscore",
                "zscore_clip": float(args.zscore_clip),
                "renderer": "flatcache_no_pycortex",
                "height": 384,
                "vmin": float(-args.zscore_clip),
                "vmax": float(args.zscore_clip),
            }
        )
        out_json.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
        if i <= 3 or i % 500 == 0:
            print(f"[{i}/{len(assigned)}] wrote {out_png.name}", flush=True)


if __name__ == "__main__":
    main()
