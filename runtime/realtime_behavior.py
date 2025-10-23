import os, time, csv, argparse
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# Local import (repo root must be on PYTHONPATH when running as a module)
from models.behavior_scorer import BehaviorScorer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or webcam index")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--minlen", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=160)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--display", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/lstm_skeleton_best.pt")
    ap.add_argument("--scores_out", default=os.environ.get("BEHAVIOR_SCORES_PATH","output/behavior_scores.csv"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pose + tracker
    model = YOLO("yolov8n-pose.pt")
    scorer = BehaviorScorer(ckpt_path=args.ckpt, threshold=args.threshold, max_len=args.maxlen)

    # CSV writer (append)
    out_csv = Path(args.scores_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    f = open(out_csv, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["ts","track_id","prob","fired","frames","note"])

    # Track buffers
    buffers = defaultdict(lambda: deque(maxlen=args.maxlen))

    # Stream predictions
    results = model.track(source=args.source, stream=True, tracker="bytetrack.yaml",
                          imgsz=args.imgsz, conf=args.conf, iou=args.iou, persist=True, verbose=False)

    for r in results:
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue
        kps_xy = r.keypoints.xy.cpu().numpy()    # (N,17,2)
        kps_cf = r.keypoints.conf.cpu().numpy()  # (N,17)
        ids    = r.boxes.id.cpu().numpy().astype(int)
        ts = time.time()

        for i, tid in enumerate(ids):
            k17x3 = np.concatenate([kps_xy[i], kps_cf[i][...,None]], axis=1)  # (17,3)
            buffers[tid].append(k17x3)
            fired = 0
            prob = 0.0
            if len(buffers[tid]) >= args.minlen:
                seq = np.stack(list(buffers[tid]), axis=0)  # (T,17,3)
                prob = float(scorer.score_track(seq))
                fired = 1 if prob >= args.threshold else 0
            w.writerow([ts, tid, f"{prob:.4f}", fired, len(buffers[tid]), ""])
            f.flush()

    f.close()

if __name__ == "__main__":
    main()
