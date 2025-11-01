#!/usr/bin/env python3
import os, shutil, pandas as pd

SRC_DEMOS = "output/demos"
FINAL = "output/final_scores.csv"
OUT_DIR = "output/top100_for_review"
K = 100

def main():
    if not os.path.exists(FINAL):
        raise SystemExit("Run tools/build_final_ensemble.py first.")
    df = pd.read_csv(FINAL)
    os.makedirs(OUT_DIR, exist_ok=True)

    copied = 0
    for _, row in df.head(K).iterrows():
        src = os.path.join(SRC_DEMOS, row["clip_path"])
        if not os.path.exists(src):
            # sometimes labeled file was from before cleanup
            continue
        dst = os.path.join(OUT_DIR, os.path.basename(row["clip_path"]))
        shutil.copy2(src, dst)
        copied += 1

    print(f"[DONE] copied {copied} clips to {OUT_DIR}")

if __name__ == "__main__":
    main()
