# FlatClip

Anonymous code release for the FlatClip flatmap pipeline.

FlatClip renders cortical fMRI activity as surface-level flatmap images, extracts
features with a frozen SigLIP2 image encoder, and trains lightweight downstream
probes. This repository contains code only. It does not include fMRI data,
rendered flatmaps, cached features, checkpoints, logs, or subject-specific
example files.

## Repository Layout

```text
scripts/
  render_hcp_dtseries_pycortex.py
  render_nsd_average_surface_fast_png.py
  render_nsd_roi_flatmaps_from_cache.py
  build_nsdgeneral_fslr32k_masks.py
  extract_flatmap_features_siglip2.py
  extract_siglip2_nsd_roi_hdf5.py
  load_siglip2.py
  run_resting_siglip_mlp_downstream.py
  train_nsd_siglip2_coco80_mlp_subjectsplit.py
  train_nsd_subjectsplit_utils.py
  extract_hcp_geometry_control_siglip2.py
  summarize_hcp_geometry_controls.py
  make_hcp_siglip_frame_count_npz.py
```

## Environment

Install the Python dependencies listed in `requirements.txt`. The exact
CUDA/PyTorch build can vary by machine.

```bash
pip install -r requirements.txt
```

SigLIP2 checkpoints are expected under `checkpoints/` by default. You can also
set a custom checkpoint root:

```bash
export FLATCLIP_CHECKPOINT_ROOT=/path/to/checkpoints
```

The default SigLIP2-NaFlex path is:

```text
$FLATCLIP_CHECKPOINT_ROOT/siglip2-base-patch16-naflex
```

## HCP Resting-State Flatmaps

Render HCP-style cortical dtseries files into frame-wise flatmap PNGs:

```bash
python scripts/render_hcp_dtseries_pycortex.py \
  --dtseries-root data/hcp/rest_dtseries \
  --filestore filestore \
  --out-dir outputs/flatmaps/hcp \
  --subjects SUBJECT_ID \
  --max-tr 40 \
  --norm-mode global-zscore \
  --scale-mode legacy \
  --height 1024 \
  --cmap inferno
```

Extract frozen SigLIP2 features from rendered flatmaps:

```bash
python scripts/extract_flatmap_features_siglip2.py \
  --model-family siglip2 \
  --variant naflex \
  --input-root outputs/flatmaps/hcp \
  --output-root outputs/features/hcp/siglip2/naflex \
  --no-resize-original \
  --save-dtype float16
```

Train the lightweight resting-state MLP probe:

```bash
python scripts/run_resting_siglip_mlp_downstream.py \
  --tasks hcp,ppmi,adni_mci,adni_ad \
  --feature-mode cls_mean_token \
  --hcp-feature-root outputs/features/hcp/siglip2/naflex \
  --ppmi-feature-root outputs/features/ppmi/siglip2/naflex \
  --adni-feature-root outputs/features/adni/siglip2/naflex \
  --hcp-roi-root data/splits/hcp/Schaefer2018_100 \
  --hcp-labels data/labels/hcp_labels.csv \
  --ppmi-roi-root data/splits/ppmi/100ROI \
  --ppmi-labels data/labels/ppmi_labels.csv \
  --adni-splits data/splits/adni \
  --out-dir outputs/resting_mlp \
  --seeds 42,43,44
```

## NSD Visual-fMRI Flatmaps

Build NSDgeneral masks in fsLR 32k space:

```bash
python scripts/build_nsdgeneral_fslr32k_masks.py \
  --nsddata data/nsd/nsddata \
  --outdir data/nsd/nsdgeneral_fslr32k_masks \
  --subjects subjXX
```

Render surface-response `.npz` files into whole-cortex or ROI flatmaps:

```bash
python scripts/render_nsd_average_surface_fast_png.py \
  --input-root data/nsd/average_surface/subjXX \
  --out-dir outputs/flatmaps/nsd/subjXX/nsdgeneral \
  --filestore filestore \
  --roi-mask data/nsd/nsdgeneral_fslr32k_masks/subjXX_nsdgeneral_fsLR32k.npz \
  --norm-mode zscore \
  --zscore-clip 3 \
  --height 384 \
  --cmap RdBu_r
```

If PyCortex flatmask and flatverts caches already exist, the cache renderer can
be used for batch rendering:

```bash
python scripts/render_nsd_roi_flatmaps_from_cache.py \
  --input-root data/nsd/average_surface/subjXX \
  --out-dir outputs/flatmaps/nsd/subjXX/nsdgeneral \
  --roi-mask data/nsd/nsdgeneral_fslr32k_masks/subjXX_nsdgeneral_fsLR32k.npz \
  --cache-dir filestore/NSD_fsLR32k/cache
```

Extract SigLIP2 patch embeddings into an HDF5 cache:

```bash
python scripts/extract_siglip2_nsd_roi_hdf5.py \
  --input-root outputs/flatmaps/nsd/subjXX/nsdgeneral \
  --output-file outputs/features/nsd/subjXX_nsdgeneral_siglip2.hdf5
```

Train the NSD COCO80 MLP probe:

```bash
python scripts/train_nsd_siglip2_coco80_mlp_subjectsplit.py \
  --hdf5-root outputs/features/nsd \
  --hdf5-template "{subject}_nsdgeneral_siglip2.hdf5" \
  --cache-root outputs/cache/nsd_siglip2_coco80 \
  --embedding-label nsdgeneral \
  --label-dir data/nsd/coco80_labels \
  --stim-info data/nsd/nsd_stim_info_merged.csv \
  --input-mode one_token \
  --one-token-source patch_mean \
  --regime within \
  --test-subject sub1 \
  --out outputs/results/nsdgeneral_sub1_seed42.json \
  --ckpt outputs/checkpoints/nsdgeneral_sub1_seed42.pt
```

## Geometry Controls

Generate SigLIP2 features for HCP flatmap perturbation controls:

```bash
python scripts/extract_hcp_geometry_control_siglip2.py \
  --variant real_flatmap \
  --input-root outputs/flatmaps/hcp \
  --output-root outputs/features/hcp_geometry_controls \
  --siglip-variant naflex
```

Supported variants include `real_flatmap`, `spatial_block_shuffle`,
`random_vertex_permutation`, `left_right_swap`,
`within_hemi_region_shuffle`, `phase_matched_randomization`,
`fc_heatmap`, and `random_image`.

Create frame-count ablation feature roots:

```bash
python scripts/make_hcp_siglip_frame_count_npz.py \
  --source-root outputs/features/hcp/siglip2/naflex \
  --output-base outputs/features/hcp_frame_counts \
  --frame-counts 10,20,40
```

Summarize geometry-control probe results:

```bash
python scripts/summarize_hcp_geometry_controls.py \
  --results-root outputs/resting_mlp_geometry_controls \
  --feature-mode cls_mean_token
```

## Notes

- Keep all data and generated artifacts under ignored directories such as
  `data/`, `outputs/`, and `checkpoints/`.
- The scripts use relative paths in examples. Pass local dataset locations via
  command-line arguments when running them.
- The repository is intentionally anonymous and code-only.
