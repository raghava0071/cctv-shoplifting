#!/usr/bin/env bash
set -euo pipefail
python tools/ucsd_frames_to_mp4.py --root datasets/ucsd/UCSD_Anomaly_Dataset.v1p2 --out datasets/ucsd_mp4 --limit 16 --fps 10
python tools/extract_pose_tracks.py --glob 'datasets/ucsd_mp4/**/*.mp4' --out pose_out_ucsd_neg
python - << 'PY'
import csv, glob
pos = glob.glob("pose_out_dcsass/**/*track_*.npy", recursive=True)
neg = glob.glob("pose_out_ucsd_neg/**/*track_*.npy", recursive=True)
with open("labels.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["path","label"])
    [w.writerow([p,"shoplifting"]) for p in pos]
    [w.writerow([n,"normal"]) for n in neg]
print("labels.csv rebuilt:", len(pos), "positives,", len(neg), "negatives")
PY
