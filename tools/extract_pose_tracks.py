import argparse, glob, os, json
from pathlib import Path
import numpy as np
import cv2

# Ultralytics YOLOv8 Pose
from ultralytics import YOLO

"""
What it does:
- Runs YOLOv8 Pose + ByteTrack on a video (or a glob of videos)
- Builds per-track pose sequences (COCO 17 keypoints -> (x,y,conf))
- Saves each track as: pose_out/{video_stem}/track_{ID}.npy with shape (T, 17, 3)
- Also writes an index JSON with meta info
"""

def process_video(video_path: Path, out_dir: Path, imgsz=640, conf=0.3, iou=0.5):
    out_dir = Path(out_dir)
    out_vid_dir = out_dir / video_path.stem
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 pose model (small & fast). You can switch to yolov8s-pose.pt if you like.
    model = YOLO("yolov8n-pose.pt")

    # Use ByteTrack for stable IDs; persist keeps IDs across frames
    results = model.track(
        source=str(video_path),
        stream=True,
        tracker="bytetrack.yaml",
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        persist=True
    )

    # Accumulate keypoints per track_id
    tracks = {}   # track_id -> list of (17, 3)
    frame_count = 0

    for r in results:
        frame_count += 1
        # r.keypoints: shape [num_dets, 17, 3] (x, y, conf)
        # r.boxes.id: track IDs [num_dets]
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue

        kps = r.keypoints.xy  # shape (N, 17, 2)
        kps_conf = r.keypoints.conf  # shape (N, 17)
        ids = r.boxes.id.cpu().numpy().astype(int)  # track IDs
        kps = kps.cpu().numpy()
        kps_conf = kps_conf.cpu().numpy()

        # combine (x,y,conf)
        for i, tid in enumerate(ids):
            xy = kps[i]          # (17, 2)
            cf = kps_conf[i]     # (17,)
            kyc = np.concatenate([xy, cf[..., None]], axis=1)  # (17, 3)
            tracks.setdefault(tid, []).append(kyc)

    # Save per-track npy
    index = {"video": str(video_path), "tracks": []}
    for tid, seq_list in tracks.items():
        seq = np.stack(seq_list, axis=0)  # (T, 17, 3)
        out_npy = out_vid_dir / f"track_{tid}.npy"
        np.save(out_npy, seq)
        index["tracks"].append({
            "track_id": int(tid),
            "frames": int(seq.shape[0]),
            "n_keypoints": int(seq.shape[1]),
            "path": str(out_npy)
        })

    # Write index JSON
    with open(out_vid_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"[OK] {video_path.name}: {len(index['tracks'])} tracks saved -> {out_vid_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, help="Path to a single video file")
    ap.add_argument("--glob", type=str, help='Glob like "data/**/*.mp4"')
    ap.add_argument("--out", type=str, default="pose_out")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    if args.video:
        videos.append(Path(args.video))
    if args.glob:
        for p in glob.glob(args.glob, recursive=True):
            if p.lower().endswith((".mp4",".mov",".avi",".mkv")):
                videos.append(Path(p))

    if not videos:
        raise SystemExit("No videos found. Use --video path.mp4 or --glob 'data/**/*.mp4'")

    for v in videos:
        process_video(v, out_dir, imgsz=args.imgsz, conf=args.conf, iou=args.iou)

if __name__ == "__main__":
    main()
