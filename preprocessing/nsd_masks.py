#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

HEMIS = {
    "lh": {"key": "left", "atlas": "L"},
    "rh": {"key": "right", "atlas": "R"},
}


@dataclass
class HemisphereSummary:
    hemi: str
    source_label: str
    native_vertices: int
    native_positive_vertices: int
    fsaverage_vertices: int
    fsaverage_positive_vertices: int
    fslr_vertices: int
    fslr_positive_vertices: int


@dataclass
class SubjectSummary:
    subject: str
    output_npz: str
    hemispheres: list[HemisphereSummary]


def parse_subjects(values: Iterable[str]) -> list[str]:
    subjects: list[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value.lower().startswith("subj"):
            subjects.append(value.lower())
        else:
            subjects.append(f"subj{int(value):02d}")
    return subjects


def normalize_sphere(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    norm = np.linalg.norm(coords, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Sphere coordinates contain zero-length vertices.")
    return coords / norm


def load_surface_vertices(path: Path) -> np.ndarray:
    data = nib.load(str(path)).agg_data()
    if isinstance(data, tuple):
        return np.asarray(data[0])
    return np.asarray(data)


def load_label(path: Path) -> np.ndarray:
    arr = np.asanyarray(nib.load(str(path)).dataobj).reshape(-1)
    return arr > 0


def build_lookup(src_sphere: Path, trg_sphere: Path) -> np.ndarray:
    src = normalize_sphere(load_surface_vertices(src_sphere))
    trg = normalize_sphere(load_surface_vertices(trg_sphere))
    return cKDTree(src).query(trg, k=1, workers=-1)[1].astype(np.int64)


def find_atlases(data_dir: Path | None) -> tuple[object, object]:
    try:
        from neuromaps import datasets
    except ImportError as exc:
        raise RuntimeError(
            "neuromaps is required to fetch the fsaverage/fsLR sphere files. "
            "Install it or pass an existing neuromaps atlas directory."
        ) from exc

    kwargs = {"verbose": 0}
    if data_dir is not None:
        kwargs["data_dir"] = str(data_dir)
    fsaverage = datasets.fetch_atlas("fsaverage", "164k", **kwargs)
    fslr = datasets.fetch_atlas("fsLR", "32k", **kwargs)
    return fsaverage, fslr


def build_subject_mask(
    nsddata: Path,
    subject: str,
    outdir: Path,
    lookups: dict[str, np.ndarray],
    fslr_medial: dict[str, np.ndarray],
) -> SubjectSummary:
    arrays: dict[str, np.ndarray] = {}
    summaries: list[HemisphereSummary] = []

    for hemi, spec in HEMIS.items():
        source_label = nsddata / "freesurfer" / subject / "label" / f"{hemi}.nsdgeneral.mgz"
        transform = nsddata / "ppdata" / subject / "transforms" / f"{hemi}.white-to-fsaverage.mgz"
        if not source_label.exists():
            raise FileNotFoundError(source_label)
        if not transform.exists():
            raise FileNotFoundError(transform)

        native = load_label(source_label)
        fsavg_to_native = np.asanyarray(nib.load(str(transform)).dataobj).reshape(-1).astype(np.int64) - 1
        if fsavg_to_native.min() < 0 or fsavg_to_native.max() >= native.shape[0]:
            raise ValueError(f"{transform} contains indices outside native label range 0..{native.shape[0] - 1}")

        fsaverage = native[fsavg_to_native]
        fslr = fsaverage[lookups[hemi]] & fslr_medial[hemi]
        fslr = np.asarray(fslr, dtype=bool)
        if fslr.shape != (32492,):
            raise ValueError(f"{subject} {hemi} yielded {fslr.shape}, expected (32492,)")

        arrays[spec["key"]] = fslr
        summaries.append(
            HemisphereSummary(
                hemi=hemi,
                source_label=str(source_label),
                native_vertices=int(native.shape[0]),
                native_positive_vertices=int(native.sum()),
                fsaverage_vertices=int(fsaverage.shape[0]),
                fsaverage_positive_vertices=int(fsaverage.sum()),
                fslr_vertices=int(fslr.shape[0]),
                fslr_positive_vertices=int(fslr.sum()),
            )
        )

    output_npz = outdir / f"{subject}_nsdgeneral_fsLR32k.npz"
    np.savez_compressed(output_npz, **arrays)
    return SubjectSummary(subject=subject, output_npz=output_npz.name, hemispheres=summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nsddata",
        type=Path,
        default=Path("<path>"),
        help="Path to the NSD nsddata directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("<path>"),
        help="Directory for generated subject .npz files and manifest.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"subj{i:02d}" for i in range(1, 9)],
        help="Subjects to process, e.g. 1 2 7 or subj01 subj02 subj07.",
    )
    parser.add_argument(
        "--neuromaps-data-dir",
        type=Path,
        default=None,
        help="Optional neuromaps data directory. Defaults to neuromaps' standard cache.",
    )
    args = parser.parse_args()

    nsddata = args.nsddata.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(args.subjects)
    fsaverage, fslr = find_atlases(args.neuromaps_data_dir)

    lookups: dict[str, np.ndarray] = {}
    fslr_medial: dict[str, np.ndarray] = {}
    lookup_paths: dict[str, str] = {}
    for hemi, spec in HEMIS.items():
        atlas_hemi = spec["atlas"]
        src_sphere = Path(getattr(fsaverage.sphere, atlas_hemi))
        trg_sphere = (
            src_sphere.parent.parent / "fsLR" / (f"tpl-fsLR_space-fsaverage_den-32k_hemi-{atlas_hemi}_sphere.surf.gii")
        )
        if not trg_sphere.exists():
            trg_sphere = Path(getattr(fslr.sphere, atlas_hemi))
        lookups[hemi] = build_lookup(src_sphere, trg_sphere)

        medial = np.asarray(nib.load(str(getattr(fslr.medial, atlas_hemi))).agg_data()).reshape(-1)
        fslr_medial[hemi] = medial > 0
        lookup_paths[f"{hemi}_source_sphere"] = str(src_sphere)
        lookup_paths[f"{hemi}_target_sphere"] = str(trg_sphere)
        lookup_paths[f"{hemi}_target_medial_wall"] = str(getattr(fslr.medial, atlas_hemi))

    np.savez_compressed(
        outdir / "fsaverage164k_to_fslr32k_nearest_lookup.npz",
        lh=lookups["lh"],
        rh=lookups["rh"],
    )

    summaries = [build_subject_mask(nsddata, subject, outdir, lookups, fslr_medial) for subject in subjects]

    manifest = {
        "nsddata": nsddata.name,
        "subjects": subjects,
        "method": (
            "official subject-native FreeSurfer nsdgeneral surface labels -> "
            "NSD white-to-fsaverage lookup -> nearest-neighbour sphere lookup "
            "from fsaverage 164k to fsLR 32k"
        ),
        "workbench_available": shutil.which("wb_command") is not None,
        "atlas_paths": {key: Path(value).name for key, value in lookup_paths.items()},
        "lookup_npz": "fsaverage164k_to_fslr32k_nearest_lookup.npz",
        "outputs": [
            {
                "subject": summary.subject,
                "output_npz": summary.output_npz,
                "hemispheres": [asdict(hemi) for hemi in summary.hemispheres],
            }
            for summary in summaries
        ],
    }

    manifest_path = outdir / "manifest_nsdgeneral_fsLR32k.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path}")
    for summary in summaries:
        counts = ", ".join(
            f"{hemi.hemi}:{hemi.fslr_positive_vertices}/{hemi.fslr_vertices}" for hemi in summary.hemispheres
        )
        print(f"{summary.subject}: {summary.output_npz} ({counts})")


if __name__ == "__main__":
    main()
