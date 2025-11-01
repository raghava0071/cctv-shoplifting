#!/usr/bin/env python3
import os, shutil
import pandas as pd

LABELS = "output/labeled_clean.csv"
SCORES = "output/frame_cnn_scores.csv"
DEMOS  = "output/demos"
OUTDIR = "output/frame_cnn_disagreements"
THR    = 0.55

def pick_score_column(df):
    for c in ["prob","score","dl_prob","pred","pred_score","shoplift_prob"]:
        if c in df.columns:
            return c
    # fallback
    for c in df.columns:
        if c not in ["file","path","source","clip_path"] and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise RuntimeError("No score column found in frame_cnn_scores.csv")

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    labels = pd.read_csv(LABELS)
    scores = pd.read_csv(SCORES)

    score_col = pick_score_column(scores)
    scores = scores[["file", score_col]].rename(columns={score_col: "dl_prob"})

    df = pd.merge(labels, scores, on="file", how="inner")
    print("[INFO] merged", len(df), "rows")

    fp = df[(df["dl_prob"] >= THR) & (df["label_int"]==0)]
    fn = df[(df["dl_prob"] <  THR) & (df["label_int"]==1)]

    copied = 0
    for _, row in pd.concat([fp, fn]).iterrows():
        src = os.path.join(DEMOS, row["file"])
        if os.path.exists(src):
            dst = os.path.join(OUTDIR, row["file"])
            shutil.copy2(src, dst)
            copied += 1

    print(f"[DONE] copied {copied} clips to {OUTDIR}")
    print(f"FP: {len(fp)}, FN: {len(fn)}")

if __name__ == "__main__":
    main()
