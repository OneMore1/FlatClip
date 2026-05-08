from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create HCP SigLIP2 feature roots with uniformly sampled frame counts."
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/features/hcp/siglip2/naflex"),
    )
    p.add_argument("--output-base", type=Path, required=True)
    p.add_argument("--frame-counts", default="5,10,20,40")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def uniform_indices(total: int, count: int) -> np.ndarray:
    if count > total:
        raise ValueError(f"Requested {count} frames from only {total}.")
    if count == total:
        return np.arange(total, dtype=np.int64)
    return np.rint(np.linspace(0, total - 1, count)).astype(np.int64)


def main() -> None:
    args = parse_args()
    counts = [int(x) for x in args.frame_counts.split(",") if x.strip()]
    files = sorted(args.source_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files under {args.source_root}")

    for count in counts:
        out_root = args.output_base / f"frames{count}" / "siglip2" / "naflex"
        out_root.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped = 0
        for src in files:
            dst = out_root / src.name
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            with np.load(src) as data:
                if "cls" not in data:
                    raise KeyError(f"{src} does not contain cls")
                cls = data["cls"].astype(np.float32, copy=False)
                idx = uniform_indices(int(cls.shape[0]), count)
                out = {
                    "cls": cls[idx].astype(np.float16),
                    "image_embeds": cls[idx].astype(np.float16),
                    "sampled_indices": idx,
                }
                if "frame_names" in data:
                    out["frame_names"] = data["frame_names"][idx]
            np.savez_compressed(dst, **out)
            meta = {
                "source": str(src),
                "frame_count": count,
                "source_frame_count": int(cls.shape[0]),
                "sampled_indices": idx.tolist(),
                "pooling_for_downstream": "run_resting_siglip_npz_mlp_downstream.py --feature-mode cls_mean_token",
            }
            dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            written += 1
        print(f"frames={count} written={written} skipped={skipped} out={out_root}", flush=True)


if __name__ == "__main__":
    main()
