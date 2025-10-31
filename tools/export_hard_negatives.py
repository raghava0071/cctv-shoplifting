#!/usr/bin/env python3
import os, re, shutil, pandas as pd, numpy as np

LABELS = "output/labels.csv"
DEMOS  = "output/demos"
OUTDIR = "output/hard_negatives"
THR = 0.55   # change if you want

if not os.path.exists(LABELS):
    raise SystemExit(f"ERROR: {LABELS} not found – run labeling first.")

df = pd.read_csv(LABELS)

def extract_prob(name: str) -> float:
    m = re.search(r"_p([0-9]+\.[0-9]+)(?=\.mp4|_)", str(name))
    return float(m.group(1)) if m else np.nan

df["prob"] = df["file"].apply(extract_prob)
df["label_int"] = df["label"].str.upper().map({"TRUE":1, "FALSE":0})

clean = df.dropna(subset=["prob","label_int"]).copy()

# model says positive
pred_pos = clean[clean["prob"] >= THR]

# human says FALSE
hard_negs = pred_pos[pred_pos["label_int"] == 0]

os.makedirs(OUTDIR, exist_ok=True)

copied = 0
for _, row in hard_negs.iterrows():
    src = os.path.join(DEMOS, row["file"])
    if os.path.exists(src):
        dst = os.path.join(OUTDIR, row["file"])
        shutil.copy2(src, dst)
        copied += 1

print("Total hard negatives:", len(hard_negs))
print(f"Copied {copied} to {OUTDIR}")
