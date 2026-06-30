# Paper Table Results

This directory contains lightweight, anonymous CSV summaries for the FlatClip
paper tables and supplementary result tables.

The bundle is intentionally text-only. It does not include fMRI data, rendered
flatmaps, subject-level feature caches, checkpoints, raw logs, host paths, or
user-specific paths.

## Files

- `table_inventory.csv`: mapping from manuscript tables/figures to bundled CSVs.
- `table1_flatclip_main.csv`: FlatClip row from the resting-state benchmark table.
  External Table 1 baseline rows are excluded by request.
- `table2_hcp_geometry_controls_siglip2.csv`: HCP geometry-control ablation with
  the SigLIP2 encoder.
- `table3_representation_scale.csv`: ROI, 4D ROI, native voxel, and adaptive
  patch representation-scale comparison.
- `table3_hcp450_roi4d_provenance.csv`: provenance-oriented HCP 450 ROI-to-4D
  check, including the manuscript value and diagnostic reruns.
- `table4_implementation_ablations.csv`: main implementation ablations for
  frozen image backbone, feature representation, and normalization.
- `supp_rest_backbone_token_ablation.csv`: supplementary backbone/token ablation.
- `supp_rest_frame_count_ablation.csv`: resting-state frame-count ablation.
- `supp_hcp_geometry_controls_dinov2.csv`: DINOv2 geometry-control robustness
  check.
- `supp_nsd_coco80_within_subject.csv`: within-subject NSD COCO80 results.
- `supp_nsd_image_fmri_alignment.csv`: stimulus-level image-fMRI alignment
  statistics.
- `nsd_flatmap_input_sizes.csv`: rendered flatmap input sizes used for NSD
  cortex, visual-cortex, and NSDgeneral inputs.
- `provenance_notes.md`: scope, exclusions, and caveats.

## Notes

All values are reported in percent when the corresponding manuscript table uses
percent units. Metrics are split into `mean` and `std` columns instead of using
formatted strings so that the files are easy to parse.
