from __future__ import annotations

from pathlib import Path

import numpy as np


def load_surface_npz(
    path: Path,
    *,
    left_key: str = "lh",
    right_key: str = "rh",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load surface arrays while skipping object-encoded metadata."""
    with np.load(path, allow_pickle=False) as data:
        left = np.asarray(data[left_key], dtype=np.float32).reshape(-1)
        right = np.asarray(data[right_key], dtype=np.float32).reshape(-1)
        metadata: dict[str, object] = {}
        for key in data.files:
            if key in (left_key, right_key):
                continue
            try:
                value = np.asarray(data[key])
            except ValueError:
                continue
            if value.dtype.hasobject:
                continue
            metadata[key] = value.tolist()
    return left, right, metadata
