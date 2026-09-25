#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import cm
from PIL import Image

if __package__:
    from .npz_io import load_surface_npz
else:
    from npz_io import load_surface_npz


def configure_cortex(filestore: Path):
    import cortex
    import cortex.database as database
    import cortex.dataset.braindata as braindata
    import cortex.quickflat.utils as qf_utils

    fs = str(filestore.resolve())
    database.default_filestore = fs
    database.db = database.Database(fs)
    database.db.reload_subjects()
    cortex.db = database.db
    braindata.db = database.db
    qf_utils.db = database.db
    return qf_utils


def discover_npz_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("sub*_*.npz"))


def load_atlasroi(info_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = info_dir / "atlasroi.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return np.asarray(data["left"], dtype=bool), np.asarray(data["right"], dtype=bool)


def load_surface_mask(path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path) as data:
        return np.asarray(data["left"], dtype=bool), np.asarray(data["right"], dtype=bool)


def combine_masks(
    *masks: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    out = None
    for mask in masks:
        if mask is None:
            continue
        left, right = mask
        out = (left.copy(), right.copy()) if out is None else (out[0] & left, out[1] & right)
    return out


def load_vector(path: Path, surface_mask: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, dict[str, object]]:
    left, right, meta = load_surface_npz(path)
    if surface_mask is not None:
        left_mask, right_mask = surface_mask
        left = left.copy()
        right = right.copy()
        left[~left_mask] = np.nan
        right[~right_mask] = np.nan
    return np.concatenate([left, right], axis=0).astype(np.float32, copy=False), meta


def zscore_vector(vec: np.ndarray) -> np.ndarray:
    finite = np.isfinite(vec)
    out = vec.astype(np.float32, copy=True)
    if not finite.any():
        return out
    mu = float(out[finite].mean())
    sigma = float(out[finite].std())
    if sigma <= 1e-8:
        out[finite] = 0.0
    else:
        out[finite] = (out[finite] - mu) / sigma
    return out


def symmetric_limits(vec: np.ndarray, percentile: float) -> tuple[float, float]:
    finite = np.isfinite(vec)
    if not finite.any():
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(vec[finite]), percentile))
    vmax = max(vmax, 1e-6)
    return -vmax, vmax


def prepare_vector(
    vec: np.ndarray, norm_mode: str, percentile: float, zscore_clip: float
) -> tuple[np.ndarray, float, float]:
    if norm_mode == "raw":
        vmin, vmax = symmetric_limits(vec, percentile)
        return vec, vmin, vmax
    if norm_mode == "zscore":
        out = zscore_vector(vec)
        finite = np.isfinite(out)
        clip = zscore_clip if zscore_clip > 0 else 3.0
        out[finite] = np.clip(out[finite], -clip, clip)
        return out, -clip, clip
    raise ValueError(f"Unsupported norm_mode: {norm_mode}")


def render_flatmap_values(vec: np.ndarray, flatmask: np.ndarray, pixmap) -> np.ndarray:
    data = vec.astype(np.float32, copy=False).ravel()
    img = np.full(flatmask.shape, np.nan, dtype=np.float32)
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    mapped = np.full(badmask.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(data)
    values = pixmap.dot(np.nan_to_num(data, nan=0.0))
    weights = pixmap.dot(valid.astype(np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        values = values / weights
    mapped[badmask] = np.asarray(values[badmask], dtype=np.float32).reshape(-1)
    img[flatmask] = mapped
    img = img.T[::-1]
    if not img.flags["C_CONTIGUOUS"]:
        img = img.copy(order="C")
    return img


def crop_rgba(rgba: np.ndarray, pad: int) -> np.ndarray:
    alpha = rgba[..., 3]
    ys, xs = np.where(alpha > 0)
    if ys.size == 0 or xs.size == 0:
        return rgba
    top = max(0, int(ys.min()) - pad)
    bottom = min(rgba.shape[0], int(ys.max()) + 1 + pad)
    left = max(0, int(xs.min()) - pad)
    right = min(rgba.shape[1], int(xs.max()) + 1 + pad)
    return rgba[top:bottom, left:right]


def values_to_rgba(img: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> np.ndarray:
    finite = np.isfinite(img)
    denom = max(vmax - vmin, 1e-6)
    norm = np.clip((np.nan_to_num(img, nan=vmin) - vmin) / denom, 0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(norm, bytes=True)
    rgba[~finite] = np.asarray([0, 0, 0, 0], dtype=np.uint8)
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--filestore", type=Path, required=True)
    parser.add_argument("--pycortex-subject", default="NSD_fsLR32k")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-pad", type=int, default=2)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--norm-mode", choices=("raw", "zscore"), default="zscore")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--zscore-clip", type=float, default=3.0)
    parser.add_argument("--roi-mask", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recache", action="store_true")
    args = parser.parse_args()

    files = discover_npz_files(args.input_root)
    if not files:
        print(f"No .npz files under {args.input_root}", file=sys.stderr)
        sys.exit(1)
    assigned = [path for index, path in enumerate(files) if index % args.world_size == args.rank]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qf_utils = configure_cortex(args.filestore)
    info_dir = args.filestore / args.pycortex_subject / "surface-info"
    surface_mask = combine_masks(load_atlasroi(info_dir), load_surface_mask(args.roi_mask))

    flatmask, _ = qf_utils.get_flatmask(args.pycortex_subject, height=args.height, recache=args.recache)
    pixmap = qf_utils.get_flatcache(args.pycortex_subject, None, height=args.height, recache=args.recache)
    print(
        f"rank={args.rank} world_size={args.world_size} assigned={len(assigned)} "
        f"height={args.height} input_root={args.input_root}",
        flush=True,
    )

    for i, path in enumerate(assigned, start=1):
        out_png = args.out_dir / f"{path.stem}.png"
        out_meta = args.out_dir / f"{path.stem}.json"
        if out_png.exists() and out_meta.exists() and not args.overwrite:
            print(f"[{i}/{len(assigned)}] skip {path.name}", flush=True)
            continue
        vec, meta = load_vector(path, surface_mask)
        vec, vmin, vmax = prepare_vector(vec, args.norm_mode, args.percentile, args.zscore_clip)
        img = render_flatmap_values(vec, flatmask, pixmap)
        rgba = values_to_rgba(img, args.cmap, vmin, vmax)
        rgba = crop_rgba(rgba, max(0, args.crop_pad))
        Image.fromarray(rgba, mode="RGBA").save(out_png)
        meta.update(
            {
                "norm_mode": args.norm_mode,
                "fast_renderer": True,
                "height": int(args.height),
                "vmin": float(vmin),
                "vmax": float(vmax),
            }
        )
        out_meta.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
        print(f"[{i}/{len(assigned)}] wrote {out_png.name}", flush=True)


if __name__ == "__main__":
    main()
