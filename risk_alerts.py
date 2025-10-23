import os, csv, time, argparse
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import torch
from ultralytics import YOLO

from models.behavior_scorer import BehaviorScorer

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--minlen", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=160)
    ap.add_argument("--min_consec", type=int, default=3, help="frames ≥ threshold needed to fire")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--display", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/lstm_skeleton_best.pt")
    ap.add_argument("--save_clips", type=int, default=0)
    ap.add_argument("--alerts_dir", default="output/alerts")
    ap.add_argument("--csv_path", default="output/alert_log.csv")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Models
    pose_model = YOLO("yolov8n-pose.pt")
    scorer = BehaviorScorer(ckpt_path=args.ckpt, threshold=args.threshold, max_len=args.maxlen)

    # I/O
    alert_csv = Path(args.csv_path)
    ensure_parent(alert_csv)
    new_file = not alert_csv.exists()
    csv_f = open(alert_csv, "a", newline="")
    csv_w = csv.writer(csv_f)
    if new_file:
        csv_w.writerow(["ts","track_id","frames","shoplift_prob","fired","source","clip_path"])

    # Per-track buffers and debounce counters
    buffers = defaultdict(lambda: deque(maxlen=args.maxlen))
    consec_over = defaultdict(int)

    results = pose_model.track(
        source=args.source, stream=True, tracker="bytetrack.yaml",
        imgsz=args.imgsz, conf=args.conf, iou=args.iou, persist=True, verbose=False
    )

    alerts_dir = Path(args.alerts_dir); alerts_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue
        ts = time.time()
        kps_xy = r.keypoints.xy.cpu().numpy()    # (N,17,2)
        kps_cf = r.keypoints.conf.cpu().numpy()  # (N,17)
        ids    = r.boxes.id.cpu().numpy().astype(int)

        for i, tid in enumerate(ids):
            k17x3 = np.concatenate([kps_xy[i], kps_cf[i][...,None]], axis=1)  # (17,3)
            buffers[tid].append(k17x3)

            prob = 0.0
            fired = 0
            if len(buffers[tid]) >= args.minlen:
                seq = np.stack(list(buffers[tid]), axis=0)
                prob = float(scorer.score_track(seq))
                if prob >= args.threshold:
                    consec_over[tid] += 1
                else:
                    consec_over[tid] = 0
                fired = 1 if consec_over[tid] >= args.min_consec else 0

            clip_path = ""
            if fired and args.save_clips:
                clip_name = f"alert_{int(ts*1000)}.txt"
                clip_path = str(alerts_dir / clip_name)
                with open(clip_path,"w") as f: f.write("alert placeholder")

            csv_w.writerow([f"{ts:.6f}", tid, len(buffers[tid]), f"{prob:.4f}", fired, args.source, clip_path])
            csv_f.flush()

    csv_f.close()

if __name__ == "__main__":
    main()
