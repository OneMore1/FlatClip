#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import numpy as np
from PIL import Image


def _configure_filestore(filestore: Path) -> None:
    import cortex.database as database

    fs = str(filestore.resolve())
    database.default_filestore = fs
    database.db = database.Database(fs)
    database.db.reload_subjects()


def _find_brain_model_axis(img: nib.Cifti2Image, n_col: int):
    hdr = img.header
    for i in range(16):
        try:
            ax = hdr.get_axis(i)
        except (IndexError, ValueError):
            continue
        size = getattr(ax, "size", None)
        if size == n_col and hasattr(ax, "vertex") and hasattr(ax, "name"):
            return ax
    return None


def load_dtseries_vertex_timeseries(
    path: Path,
    n_lh: int,
    n_rh: int,
) -> np.ndarray:
    """
    Load a .dtseries.nii and return array (n_lh + n_rh, n_tr), cortical vertices only
    placed by BrainModelAxis vertex id; medial-wall vertices stay NaN.
    """
    img = nib.load(str(path))
    raw = np.asarray(img.get_fdata(dtype=np.float64), dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D data in {path}, got shape {raw.shape}")

    series = raw if raw.shape[0] <= raw.shape[1] else raw.T
    n_tr, n_g = series.shape

    bm = _find_brain_model_axis(img, n_g)
    if bm is None:
        raise RuntimeError(f"Could not find BrainModelAxis with size {n_g} in {path}")

    names = np.asarray(bm.name)
    verts = np.asarray(bm.vertex, dtype=np.int64)
    out = np.full((n_lh + n_rh, n_tr), np.nan, dtype=np.float64)

    sn = np.asarray([str(x) for x in names])
    ml = sn == "CORTEX_LEFT"
    mr = sn == "CORTEX_RIGHT"
    if not np.any(ml):
        ml = sn == "CIFTI_STRUCTURE_CORTEX_LEFT"
    if not np.any(mr):
        mr = sn == "CIFTI_STRUCTURE_CORTEX_RIGHT"
    if not np.any(ml) or not np.any(mr):
        raise RuntimeError(
            f"{path}: missing cortex columns (expected CORTEX_LEFT/RIGHT or "
            f"CIFTI_STRUCTURE_CORTEX_LEFT/RIGHT); got sample {np.unique(sn)[:8]}"
        )

    vl, vr = verts[ml], verts[mr]
    if vl.min() < 0 or vl.max() >= n_lh or vr.min() < 0 or vr.max() >= n_rh:
        raise RuntimeError(
            f"{path}: vertex id out of range for n_lh={n_lh} n_rh={n_rh} "
            f"(L {int(vl.min())}..{int(vl.max())}, R {int(vr.min())}..{int(vr.max())})"
        )

    out[vl, :] = series[:, ml].T
    out[n_lh + vr, :] = series[:, mr].T
    return out


def discover_dtseries_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def _output_tag_from_stem(stem: str) -> str:
    """e.g. SUBJECT_REST1_LR_Atlas_MSMAll_hp2000_clean -> SUBJECT_REST1_LR"""
    m = re.match(r"^(\d+)_(REST\d+_[A-Z]{2})", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return stem


def _framewise_limits(vec: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(vec)
    if not finite.any():
        return 0.0, 1.0
    v = vec[finite]
    vmax = float(np.percentile(v, 99.0))
    vmin = float(np.percentile(v, 1.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _global_limits(mat: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(mat)
    if not finite.any():
        return 0.0, 1.0
    v = mat[finite]
    vmax = float(np.percentile(v, 99.0))
    vmin = float(np.percentile(v, 1.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _percentile_limits_from_values(
    values: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(values, high_pct))
    vmin = float(np.percentile(values, low_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _global_spatiotemporal_zscore(mat: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(mat)
    if not finite.any():
        return np.asarray(mat, dtype=np.float64), 0.0, 1.0
    v = mat[finite]
    mu = float(np.mean(v))
    sigma = float(np.std(v))
    if sigma <= 0:
        sigma = 1.0
    out = np.asarray(mat, dtype=np.float64).copy()
    out[finite] = (out[finite] - mu) / sigma
    return out, mu, sigma


def _voxel_temporal_zscore(mat: np.ndarray) -> np.ndarray:
    """
    Z-score each vertex (row) over time (columns), ignoring NaNs.

    Input shape: (n_vertices, n_tr_selected)
    """
    out = np.asarray(mat, dtype=np.float64).copy()
    finite = np.isfinite(out)

    # Compute per-vertex stats only on finite samples.
    counts = np.sum(finite, axis=1).astype(np.float64)
    den = np.maximum(counts, 1.0)
    sums = np.where(finite, out, 0.0).sum(axis=1)
    means = np.zeros_like(sums, dtype=np.float64)
    np.divide(sums, den, out=means, where=den > 0)

    centered = np.where(finite, out - means[:, None], 0.0)
    sq_sums = np.sum(centered * centered, axis=1)
    vars_ = np.zeros_like(sq_sums, dtype=np.float64)
    np.divide(sq_sums, den, out=vars_, where=den > 0)
    stds = np.sqrt(vars_)
    stds = np.where(stds > 0, stds, 1.0)

    out = np.where(finite, centered / stds[:, None], np.nan)
    return out


def _apply_norm_mode(
    seg: np.ndarray,
    norm_mode: str,
    fp_name: str,
) -> tuple[np.ndarray, str | None]:
    seg = np.asarray(seg, dtype=np.float64)
    if norm_mode == "global-zscore":
        seg, mu, sigma = _global_spatiotemporal_zscore(seg)
        return seg, f"Norm global-zscore for {fp_name}: mu={mu:.6f} sigma={sigma:.6f}"
    if norm_mode == "voxel-zscore":
        seg = _voxel_temporal_zscore(seg)
        return seg, f"Norm voxel-zscore for {fp_name}"
    return seg, None


def _sample_finite_values(mat: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    vals = np.asarray(mat[np.isfinite(mat)], dtype=np.float64)
    if vals.size == 0:
        return vals
    if max_n > 0 and vals.size > max_n:
        idx = rng.choice(vals.size, size=max_n, replace=False)
        vals = vals[idx]
    return vals


def _alpha_bbox(path: Path) -> tuple[int, int, int, int] | None:
    im = Image.open(path).convert("RGBA")
    alpha = np.array(im.getchannel("A"))
    ys, xs = np.where(alpha > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop_all_frames_same_box(frame_paths: list[Path], pad: int = 6) -> None:
    boxes = []
    for p in frame_paths:
        b = _alpha_bbox(p)
        if b is not None:
            boxes.append(b)
    if not boxes:
        return
    left = max(b[0] for b in boxes)
    top = max(b[1] for b in boxes)
    right = min(b[2] for b in boxes)
    bottom = min(b[3] for b in boxes)
    if right <= left or bottom <= top:
        return
    left = max(0, left - pad)
    top = max(0, top - pad)
    for p in frame_paths:
        im = Image.open(p)
        w, h = im.size
        r = min(w, right + pad)
        b = min(h, bottom + pad)
        if r > left and b > top:
            im.crop((left, top, r, b)).save(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dtseries-root",
        type=Path,
        default=Path("<path>"),
        help="Root folder to search for *.dtseries.nii.",
    )
    ap.add_argument(
        "--glob",
        dest="glob_pat",
        default="*.dtseries.nii",
        help="Glob relative to --dtseries-root (recursive rglob)",
    )
    ap.add_argument(
        "--filestore",
        type=Path,
        default=Path("<path>"),
        help="PyCortex filestore (same as GIFTI renderer)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("<path>"),
        help="Output root: <out-dir>/<tag>/frame_XXXXXX.png",
    )
    ap.add_argument("--pycortex-subject", default="HCP_S1200_32k")
    ap.add_argument(
        "--files",
        default="",
        help="Comma-separated explicit .dtseries.nii paths; if set, ignores --dtseries-root discovery",
    )
    ap.add_argument(
        "--subjects",
        default="",
        help="Comma-separated HCP IDs; keep only files whose stem starts with <id>_",
    )
    ap.add_argument("--n-vertices-lh", type=int, default=32492)
    ap.add_argument("--n-vertices-rh", type=int, default=32492)
    ap.add_argument("--start-tr", type=int, default=0)
    ap.add_argument(
        "--max-tr",
        type=int,
        default=40,
        help="Max TRs per file from --start-tr (default 40). Use -1 for all TRs.",
    )
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--recache", action="store_true")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument(
        "--norm-mode",
        choices=["framewise", "global-zscore", "voxel-zscore"],
        default="framewise",
        help=(
            "Normalization mode for values passed to pycortex. "
            "framewise = per-frame percentile limits (legacy behavior); "
            "global-zscore = z-score over all selected TRs and vertices, then "
            "use one global percentile range for all frames in a file; "
            "voxel-zscore = z-score each vertex over selected TRs, then "
            "use one global percentile range for all frames in a file."
        ),
    )
    ap.add_argument(
        "--scale-mode",
        choices=["legacy", "per-file", "all-files"],
        default="legacy",
        help=(
            "How vmin/vmax are selected. "
            "legacy = framewise keeps per-frame limits, other norm modes keep per-file limits; "
            "per-file = one percentile range per file for all modes; "
            "all-files = one shared percentile range across all selected files (subjects)."
        ),
    )
    ap.add_argument(
        "--pct-low",
        type=float,
        default=1.0,
        help="Lower percentile for color limits (default 1.0)",
    )
    ap.add_argument(
        "--pct-high",
        type=float,
        default=99.0,
        help="Upper percentile for color limits (default 99.0)",
    )
    ap.add_argument(
        "--scale-sample-per-file",
        type=int,
        default=20000,
        help=(
            "When --scale-mode all-files, max finite values sampled per file "
            "for global percentile estimation (default 20000)."
        ),
    )
    ap.add_argument(
        "--scale-seed",
        type=int,
        default=42,
        help="RNG seed for all-files scale sampling",
    )
    ap.add_argument(
        "--fixed-vmin",
        type=float,
        default=None,
        help="If set with --fixed-vmax, force one exact color range for all frames/files.",
    )
    ap.add_argument(
        "--fixed-vmax",
        type=float,
        default=None,
        help="If set with --fixed-vmin, force one exact color range for all frames/files.",
    )
    ap.add_argument(
        "--scale-only",
        action="store_true",
        help=(
            "Only compute and print scale info, then exit. "
            "Useful with --scale-mode all-files to precompute one shared range."
        ),
    )
    args = ap.parse_args()

    if args.pct_high <= args.pct_low:
        print("Invalid percentiles: require --pct-high > --pct-low", file=sys.stderr)
        sys.exit(1)

    fixed_scale = (args.fixed_vmin is not None) or (args.fixed_vmax is not None)
    if fixed_scale and (args.fixed_vmin is None or args.fixed_vmax is None):
        print("Set both --fixed-vmin and --fixed-vmax together.", file=sys.stderr)
        sys.exit(1)
    if fixed_scale and args.fixed_vmax <= args.fixed_vmin:
        print("Invalid fixed range: require --fixed-vmax > --fixed-vmin", file=sys.stderr)
        sys.exit(1)

    if args.files.strip():
        paths = [Path(s.strip()) for s in args.files.split(",") if s.strip()]
        for p in paths:
            if not p.is_file():
                print(f"Missing file: {p}", file=sys.stderr)
                sys.exit(1)
    else:
        if not args.dtseries_root.is_dir():
            print(f"Not a directory: {args.dtseries_root}", file=sys.stderr)
            sys.exit(1)
        paths = discover_dtseries_files(args.dtseries_root, args.glob_pat)
    if not paths:
        print("No dtseries files to process.", file=sys.stderr)
        sys.exit(1)

    if args.subjects.strip():
        allow = {s.strip() for s in args.subjects.split(",") if s.strip()}
        filt = []
        for p in paths:
            stem = p.stem.replace(".dtseries", "")
            ok = any(stem.startswith(f"{sid}_") for sid in allow)
            if ok:
                filt.append(p)
        paths = filt
    if not paths:
        print("No files left after --subjects filter.", file=sys.stderr)
        sys.exit(1)

    _configure_filestore(args.filestore)

    import cortex
    import cortex.database as database

    fs = str(args.filestore.resolve())
    database.default_filestore = fs
    database.db = database.Database(fs)
    database.db.reload_subjects()
    cortex.db = database.db
    import cortex.dataset.braindata as braindata

    braindata.db = database.db
    import cortex.quickflat.utils as qf_utils

    qf_utils.db = database.db

    if not hasattr(cortex, "quickflat"):
        print("PyCortex quickflat not available", file=sys.stderr)
        sys.exit(1)

    n_lh, n_rh = args.n_vertices_lh, args.n_vertices_rh

    shared_vmin = shared_vmax = None
    if fixed_scale:
        shared_vmin = float(args.fixed_vmin)
        shared_vmax = float(args.fixed_vmax)
        print(f"Using fixed color scale for all files: vmin={shared_vmin:.6f} vmax={shared_vmax:.6f}")
    elif args.scale_mode == "all-files":
        rng = np.random.default_rng(args.scale_seed)
        samples: list[np.ndarray] = []
        for fp in paths:
            try:
                data = load_dtseries_vertex_timeseries(fp, n_lh=n_lh, n_rh=n_rh)
            except (OSError, RuntimeError, ValueError) as e:
                print(f"Skip scale-prepass {fp}: {e}", file=sys.stderr)
                continue

            n_tr = data.shape[1]
            t0 = max(0, args.start_tr)
            t1 = n_tr if args.max_tr < 0 else min(n_tr, t0 + args.max_tr)
            seg = np.asarray(data[:, t0:t1], dtype=np.float64)
            seg, _ = _apply_norm_mode(seg, args.norm_mode, fp.name)
            s = _sample_finite_values(seg, args.scale_sample_per_file, rng)
            if s.size > 0:
                samples.append(s)

        if not samples:
            print(
                "Could not compute all-files scale: no finite sampled values.",
                file=sys.stderr,
            )
            sys.exit(1)

        pooled = np.concatenate(samples, axis=0)
        shared_vmin, shared_vmax = _percentile_limits_from_values(pooled, args.pct_low, args.pct_high)
        print(
            f"Using shared all-files color scale: vmin={shared_vmin:.6f} vmax={shared_vmax:.6f} "
            f"(samples={pooled.size}, files={len(samples)})"
        )

    if args.scale_only:
        if shared_vmin is None or shared_vmax is None:
            print(
                "--scale-only requires either --scale-mode all-files or --fixed-vmin/--fixed-vmax",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"GLOBAL_SCALE vmin={shared_vmin:.10f} vmax={shared_vmax:.10f}")
        return

    for fp in paths:
        tag = _output_tag_from_stem(fp.stem.replace(".dtseries", ""))
        try:
            data = load_dtseries_vertex_timeseries(fp, n_lh=n_lh, n_rh=n_rh)
        except (OSError, RuntimeError, ValueError) as e:
            print(f"Skip {fp}: {e}", file=sys.stderr)
            continue

        n_tr = data.shape[1]
        t0 = max(0, args.start_tr)
        t1 = n_tr if args.max_tr < 0 else min(n_tr, t0 + args.max_tr)

        out_subj = args.out_dir / tag
        out_subj.mkdir(parents=True, exist_ok=True)
        rendered: list[Path] = []

        seg = np.asarray(data[:, t0:t1], dtype=np.float64)
        seg, norm_msg = _apply_norm_mode(seg, args.norm_mode, fp.name)
        if norm_msg:
            print(norm_msg)

        file_vmin = file_vmax = None
        if shared_vmin is not None and shared_vmax is not None:
            file_vmin, file_vmax = shared_vmin, shared_vmax
        elif args.scale_mode == "per-file" or (args.scale_mode == "legacy" and args.norm_mode != "framewise"):
            file_vmin, file_vmax = _percentile_limits_from_values(seg[np.isfinite(seg)], args.pct_low, args.pct_high)

        if file_vmin is not None and file_vmax is not None:
            print(f"Color scale for {fp.name}: vmin={file_vmin:.6f} vmax={file_vmax:.6f} (mode={args.scale_mode})")

        for t in range(t0, t1):
            vec = np.asarray(seg[:, t - t0], dtype=float)
            if not np.any(np.isfinite(vec)):
                continue
            if file_vmin is not None and file_vmax is not None:
                vmin_f, vmax_f = file_vmin, file_vmax
            else:
                vmin_f, vmax_f = _framewise_limits(vec)
            vbd = cortex.Vertex(
                vec,
                args.pycortex_subject,
                cmap=args.cmap,
                vmin=vmin_f,
                vmax=vmax_f,
            )
            out_png = out_subj / f"frame_{t:06d}.png"
            cortex.quickflat.make_png(
                str(out_png),
                vbd,
                recache=args.recache,
                height=args.height,
                with_colorbar=False,
                with_rois=False,
                with_labels=False,
                with_dropout=False,
            )
            rendered.append(out_png)

        _crop_all_frames_same_box(rendered, pad=6)
        print(f"Wrote {len(rendered)} frames under {out_subj} (from {fp.name})")


if __name__ == "__main__":
    main()
