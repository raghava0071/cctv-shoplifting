import argparse, time, os
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
from ultralytics import YOLO
from models.behavior_scorer import BehaviorScorer

def mine_from_video(video_path: Path, out_dir: Path, scorer: BehaviorScorer, minlen=32, maxlen=160, neg_thresh=0.20, imgsz=640, conf=0.25, iou=0.5):
    model = YOLO("yolov8n-pose.pt")
    # Track with ByteTrack; stream frames
    results = model.track(source=str(video_path), stream=True, tracker="bytetrack.yaml",
                          imgsz=imgsz, conf=conf, iou=iou, persist=True, verbose=False)

    # per-track pose buffers & running max prob
    buffers = defaultdict(lambda: deque(maxlen=maxlen))
    maxprob = defaultdict(lambda: 0.0)
    saved = 0

    for r in results:
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue
        kps_xy = r.keypoints.xy.cpu().numpy()    # (N,17,2)
        kps_cf = r.keypoints.conf.cpu().numpy()  # (N,17)
        ids    = r.boxes.id.cpu().numpy().astype(int)

        # update buffers and score each track
        for i, tid in enumerate(ids):
            k17x3 = np.concatenate([kps_xy[i], kps_cf[i][...,None]], axis=1)  # (17,3)
            buffers[tid].append(k17x3)

            # Only score if we have enough frames; keep running max
            if len(buffers[tid]) >= minlen:
                seq = np.stack(list(buffers[tid]), axis=0)  # (T,17,3)
                p = float(scorer.score_track(seq))
                if p > maxprob[tid]:
                    maxprob[tid] = p

    # After stream ends: dump tracks that look normal throughout
    vid_stem = video_path.stem
    dest_dir = out_dir / vid_stem
    dest_dir.mkdir(parents=True, exist_ok=True)

    for tid, buf in buffers.items():
        if len(buf) >= minlen and maxprob[tid] < neg_thresh:
            arr = np.stack(list(buf), axis=0)  # (T,17,3)
            np.save(dest_dir / f"track_{tid:04d}.npy", arr)
            saved += 1
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="datasets/dcsass/**/*.mp4")
    ap.add_argument("--out", default="pose_out_neg_mined")
    ap.add_argument("--ckpt", default="runs/lstm_skeleton_best.pt")
    ap.add_argument("--minlen", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=160)
    ap.add_argument("--neg_thresh", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    args = ap.parse_args()

    scorer = BehaviorScorer(ckpt_path=args.ckpt, threshold=0.5, max_len=args.maxlen)
    vids = sorted(Path(".").glob(args.glob), key=lambda p: p.name)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for v in vids:
        try:
            n = mine_from_video(v, out_dir, scorer, args.minlen, args.maxlen, args.neg_thresh, args.imgsz, args.conf, args.iou)
            print(f"[OK] {v.name}: {n} negative tracks saved")
            total_saved += n
        except Exception as e:
            print(f"[WARN] {v}: {e}")
    print(f"Done. Total negative tracks saved: {total_saved}")

if __name__ == "__main__":
    main()
