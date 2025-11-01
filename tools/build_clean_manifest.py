#!/usr/bin/env python3
import os, glob, cv2, pandas as pd

DEMOS = "output/demos"
OUT_CSV = "output/demos_clean.csv"

rows = []
good = 0
bad = 0

for path in sorted(glob.glob(os.path.join(DEMOS, "*.mp4"))):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        bad += 1
        cap.release()
        continue
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames <= 0:
        bad += 1
        continue
    rows.append({
        "file": os.path.basename(path),
        "path": path,
        "frames": frames,
    })
    good += 1

os.makedirs("output", exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"[DONE] wrote {good} good clips to {OUT_CSV}, skipped {bad} bad ones")
