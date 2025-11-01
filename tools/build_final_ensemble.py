#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np

OUT_DIR = "output"
LABELED = os.path.join(OUT_DIR, "labeled_clean.csv")
FRAME_SCORES = os.path.join(OUT_DIR, "frame_cnn_scores.csv")
VIDEO_SCORES = os.path.join(OUT_DIR, "video_lstm_scores.csv")
OUT_CSV = os.path.join(OUT_DIR, "final_scores.csv")

def extract_prob_from_name(name: str) -> float:
    m = re.search(r"_p([0-9]+\.[0-9]+)(?=\\.mp4|_)", str(name))
    return float(m.group(1)) if m else np.nan

def load_scores(path: str, wanted_cols=("score","prob","frame_score","video_score")):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["clip_path","score"])
    df = pd.read_csv(path)
    # normalize column name
    col = None
    for c in df.columns:
        lc = c.lower()
        if any(w in lc for w in wanted_cols):
            col = c
            break
    if col is None:
        raise RuntimeError(f"Could not find a score column in {path}. Found columns: {df.columns.tolist()}")
    # normalize clip path col
    if "clip_path" not in df.columns:
        if "file" in df.columns:
            df = df.rename(columns={"file": "clip_path"})
        elif "path" in df.columns:
            df = df.rename(columns={"path": "clip_path"})
    return df[["clip_path", col]].rename(columns={col: "score"})

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) labeled (human) data
    if not os.path.exists(LABELED):
        raise SystemExit("No output/labeled_clean.csv found. Run your labeling -> clean step first.")
    labeled = pd.read_csv(LABELED)
    # normalize name
    labeled["clip_path"] = labeled["file"]
    labeled["yolo_prob"] = labeled["file"].apply(extract_prob_from_name)

    # 2) frame cnn
    frame_df = load_scores(FRAME_SCORES)
    frame_df = frame_df.rename(columns={"score": "frame_cnn_score"})

    # 3) video lstm
    video_df = load_scores(VIDEO_SCORES)
    video_df = video_df.rename(columns={"score": "video_lstm_score"})

    # 4) merge
    df = labeled.merge(frame_df, on="clip_path", how="left")
    df = df.merge(video_df, on="clip_path", how="left")

    # fill missing scores with 0 (some clips might not have DL score)
    for col in ["yolo_prob","frame_cnn_score","video_lstm_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # 5) final ensemble (you can tune weights)
    w_yolo = 0.4
    w_frame = 0.3
    w_video = 0.3
    df["final_score"] = (
        w_yolo * df["yolo_prob"].astype(float)
        + w_frame * df["frame_cnn_score"].astype(float)
        + w_video * df["video_lstm_score"].astype(float)
    )

    # keep nice columns
    keep = [
        "clip_path",
        "file",
        "label",
        "yolo_prob",
        "frame_cnn_score",
        "video_lstm_score",
        "final_score",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("final_score", ascending=False)

    df.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {len(df)} rows to {OUT_CSV}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
