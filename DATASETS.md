# Datasets (local only)

This repo does **not** contain full CCTV / DCSASS videos because of size and licensing.

Put your data like this on your machine:

datasets/
└── dcsass/
    └── DCSASS Dataset/
        └── Shoplifting/*.mp4

Then run:
  python tools/eval_thresholds.py
  python tools/export_hard_negatives.py

The scripts will read from:
- output/labels.csv          (your human labels)
- output/demos/*.mp4         (generated clips)
