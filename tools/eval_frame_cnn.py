#!/usr/bin/env python3
"""
Evaluate the SECOND-STAGE DL model (ResNet18) on human-labeled clips.

- input 1: output/labeled_clean.csv  (714 rows, has file, label, label_int, prob_from_filename)
- input 2: output/frame_cnn_scores.csv (3710 rows, from tools/run_frame_cnn_on_demos.py)
  NOTE: this file may NOT use the column name "prob", so we auto-detect it.
"""
import os
import pandas as pd
import numpy as np

LABELS_CSV = "output/labeled_clean.csv"
SCORES_CSV = "output/frame_cnn_scores.csv"
OUT_CSV    = "output/frame_cnn_eval.csv"

# columns we will try in the scores file
CANDIDATE_SCORE_COLS = [
    "prob",
    "score",
    "dl_prob",
    "pred",
    "pred_score",
    "shoplift_prob",
]

def pick_score_column(df_scores: pd.DataFrame) -> str:
    """Try to find the best column to use as the DL score."""
    cols = list(df_scores.columns)
    print("[INFO] frame_cnn_scores.csv columns:", cols)

    # 1) try known names
    for c in CANDIDATE_SCORE_COLS:
        if c in df_scores.columns:
            print(f"[INFO] using score column: {c}")
            return c

    # 2) otherwise pick the first numeric-looking column that is not file/path
    skip = {"file", "path", "source", "clip_path", "name"}
    for c in df_scores.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df_scores[c]):
            print(f"[INFO] auto-picked numeric column: {c}")
            return c

    raise RuntimeError("Could not find a numeric score column in output/frame_cnn_scores.csv")

def main():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing {LABELS_CSV}")
    if not os.path.exists(SCORES_CSV):
        raise FileNotFoundError(f"Missing {SCORES_CSV}")

    labels = pd.read_csv(LABELS_CSV)
    scores = pd.read_csv(SCORES_CSV)

    score_col = pick_score_column(scores)

    # keep only what we need from scores
    scores_small = scores[["file", score_col]].copy()
    scores_small = scores_small.rename(columns={score_col: "dl_prob"})

    # merge
    df = pd.merge(labels, scores_small, on="file", how="inner")
    print(f"[INFO] merged rows: {len(df)}")

    y_true  = df["label_int"].values
    y_score = df["dl_prob"].values

    rows = []
    for thr in np.arange(0.30, 0.90, 0.02):
        y_pred = (y_score >= thr).astype(int)
        tp = int(((y_pred==1)&(y_true==1)).sum())
        fp = int(((y_pred==1)&(y_true==0)).sum())
        tn = int(((y_pred==0)&(y_true==0)).sum())
        fn = int(((y_pred==0)&(y_true==1)).sum())
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        acc  = (tp+tn) / len(df) if len(df)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec+1e-6)
        rows.append(dict(thr=round(thr,2), prec=prec, rec=rec, acc=acc,
                         f1=f1, tp=tp, fp=fp, tn=tn, fn=fn))

    res = pd.DataFrame(rows)
    os.makedirs("output", exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    print("\nTop by F1:")
    print(res.sort_values("f1", ascending=False).head(10).to_string(index=False))
    print(f"\n[OK] wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
