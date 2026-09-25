# FlatClip

Core pipeline: cortical surface signals to flatmap images, frozen SigLIP2
features, and downstream probes.

```text
preprocessing/
  hcp_flatmaps.py       # CIFTI-to-flatmap rendering
  nsd_flatmaps.py       # NSD surface-to-image rendering
  nsd_masks.py          # NSDgeneral surface masks
  nsd_roi_flatmaps.py   # Cached ROI-mask rendering
  npz_io.py             # Surface-array loading
inference/
  siglip2.py            # Frozen image encoder
  rest_features.py     # Resting-state feature extraction
  nsd_features.py      # NSD feature extraction
  rest_probe.py        # Resting-state downstream probe
  nsd_probe.py         # NSD COCO80 downstream probe
  nsd_utils.py         # NSD labels, splits, and utilities
```

Install Python 3.10+ dependencies with `pip install -r requirements.txt`.
Run entry points from the repository root, for example
`python -m preprocessing.hcp_flatmaps --help` or
`python -m inference.rest_features --help`.

All `<path>` values are placeholders: supply dataset, surface/flatmap cache,
checkpoint, label, split, and output locations through the script arguments.
For the shared encoder loader, `FLATCLIP_CHECKPOINT_ROOT` points directly to
the local SigLIP2 checkpoint directory (including its model and processor
files). The NSD extractor also accepts `--checkpoint`.

Preprocessing requires registered cortical surfaces and a configured Pycortex
filestore. Whole-cortex NSD rendering uses `nsd_flatmaps.py`; masked rendering
uses a supplied visual/NSDgeneral mask. The cached ROI renderer also requires
the Pycortex flatmask and vertex-projection cache.

This release contains no ablation runners, datasets, subject lists, generated
features, or model weights.
