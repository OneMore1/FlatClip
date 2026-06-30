# Provenance Notes

This result bundle is designed for an anonymous public code release.

## Scope

- Included: lightweight CSV summaries needed to recreate the paper's main and
  supplementary result tables.
- Excluded: raw fMRI data, rendered flatmaps, subject-level feature caches,
  checkpoints, raw logs, hostnames, user paths, and private storage locations.
- Excluded by request: external baseline rows from the main resting-state
  benchmark table.

## 4D ROI Volume Notes

The representation-scale table reports the manuscript values for ROI time
series, 4D ROI volumes, native voxel volumes, and adaptive patching.

For the HCP 450 ROI-to-4D row, the closest reproduction used the original
APT-style route:

1. map 450 ROI time series into an atlas-defined 4D volume by assigning each
   voxel the mean time series of its ROI;
2. use the APT foreground/multiscale patch tokenizer rather than a dense
   full-grid patch tokenizer;
3. select the test result at the best validation AUC epoch across three folds.

The manuscript row is `80.47 +/- 6.44` ACC and `80.42 +/- 6.46` weighted F1.
A live diagnostic rerun of the same route produced `81.15 +/- 4.15` ACC and
`81.09 +/- 4.33` weighted F1, which is included only as a provenance check.

## Caveats

- The bundled CSVs are table-level summaries, not raw per-subject predictions.
- Some supplementary tables include evaluated baselines because they are part of
  those supplementary result tables; the explicit exclusion only applies to the
  external baseline rows in the main resting-state benchmark table.
- Values are stored as numeric mean/std columns. Formatting such as boldface,
  heatmaps, and significance markers should be applied by downstream plotting or
  manuscript code.
