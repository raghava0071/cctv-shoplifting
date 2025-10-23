# Datasets & Provenance

## External sources (do **not** commit raw videos)
- **DCSASS Dataset** (Kaggle): `mateohervas/dcsass-dataset` — License: CC-BY-NC-SA-4.0  
  Used subset: `Shoplifting/**/*.mp4`
- **UCSD Anomaly Detection v1p2** (Kaggle mirror `karthiknm1/ucsd-anomaly-detection-dataset`)  
  Used subset converted to MP4 with our script: `UCSDped1/Test/* -> sequence.mp4` (first 16 folders)

## Derived artifacts we *do* commit
- `labels.csv` (track-level labels: `shoplifting` vs `normal`)
- `output/alert_log.csv` (runtime row-per-frame with `shoplift_prob`)
- `output/alert_events.json` (merged events)
- `runs/lstm_skeleton_best.pt` (trained LSTM)
- Manifests (paths, sizes, checksums) for:
  - `pose_out_dcsass/**/track_*.npy` (positives)
  - `pose_out_ucsd_neg/**/track_*.npy` (negatives)
  - `datasets/ucsd_mp4/**/sequence.mp4` (converted small MP4s), if small enough

> We do **not** commit Kaggle videos due to size/licensing. Use the scripts below to fetch/convert on any machine.
