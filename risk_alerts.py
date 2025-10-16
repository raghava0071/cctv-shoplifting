import os, sys, cv2, math, time, csv
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "data/test_store_clip.avi"
ALERT_DIR  = "output/alerts"
LOG_PATH   = "output/alert_log.csv"

os.makedirs("output", exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)

# Zones (percent of width): left = "shelf", right = "exit"
SHELF_X_PCT = 0.30
EXIT_X_PCT  = 0.80

# Heuristics / thresholds
PRE_SEC  = 1.0
POST_SEC = 1.0
RISK_THRESHOLD = 2.0            # adjust as needed
MIN_VISIBLE_SEC_FOR_ALERT = 0.5
ALERT_COOLDOWN_SEC        = 3.0

# Create CSV with header once
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_unix","frame","track_id","risk","item","occluded","speed","video"])

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aw, ah = max(0, ax2 - ax1), max(0, ay2 - ay1)
    bw, bh = max(0, bx2 - bx1), max(0, by2 - by1)
    union = aw*ah + bw*bh - inter + 1e-6
    return inter / union

def save_clip(frames, fps):
    ts = int(time.time()*1000)
    path = os.path.join(ALERT_DIR, f"alert_{ts}.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  -> saved {path}")
    return path

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: {VIDEO_PATH} missing"); sys.exit(1)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open video"); sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shelf_x = int(SHELF_X_PCT * W)
    exit_x  = int(EXIT_X_PCT  * W)

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    pre_frames  = int(PRE_SEC * fps)
    post_frames = int(POST_SEC * fps)
    ring = deque(maxlen=pre_frames)
    pending_after = 0
    clip_frames = []

    track_meta = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        ring.append(frame.copy())

        # Detection
        res = model(frame, conf=0.35, verbose=False)[0]
        dets, objs = [], []
        for box, cls_idx, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
            cls = model.model.names[int(cls_idx)]
            x1, y1, x2, y2 = map(int, box.tolist())
            dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), cls))
            objs.append(([x1, y1, x2, y2], cls))

        # Tracking
        tracks = tracker.update_tracks(dets, frame=frame)

        # Draw zones
        cv2.line(frame, (shelf_x, 0), (shelf_x, H), (255, 255, 255), 1)
        cv2.line(frame, (exit_x, 0), (exit_x, H), (255, 255, 255), 1)

        alert_now = False

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            meta = track_meta.setdefault(tid, {
                "frames_visible": 0,
                "item": 0,
                "occluded": 0,
                "last_centroid": (cx, cy),
                "speed": 0.0,
                "seen_in_shelf_zone": 0,
                "last_alert_frame": -10**9
            })
            meta["frames_visible"] += 1

            # Item overlap
            track_box = (x1, y1, x2, y2)
            for (bx1, by1, bx2, by2), c in objs:
                if c in ("backpack", "handbag", "suitcase", "bottle", "cell phone"):
                    if iou(track_box, (bx1, by1, bx2, by2)) > 0.10:
                        meta["item"] = 1

            # Occlusion proxy
            if getattr(t, "time_since_update", 0) > 0:
                meta["occluded"] += 1

            # Speed
            lx, ly = meta["last_centroid"]
            speed = math.hypot(cx - lx, cy - ly)
            meta["speed"] = speed
            meta["last_centroid"] = (cx, cy)

            # Zones
            if cx <= shelf_x:
                meta["seen_in_shelf_zone"] = 1
            in_exit = cx >= exit_x

            # Risk (tuned)
            risk = 0.0
            risk += 1.4 * meta["item"]
            risk += 0.3 * min(meta["occluded"], 5)
            if meta["speed"] > 4.0:
                risk += 0.4
            if meta["seen_in_shelf_zone"] and in_exit and meta["frames_visible"] > int(0.5 * fps):
                risk += 0.8

            # Overlay
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid} R:{risk:.2f}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Alert gating + cooldown
            min_visible_frames = int(MIN_VISIBLE_SEC_FOR_ALERT * fps)
            cooldown_frames = int(ALERT_COOLDOWN_SEC * fps)
            if (
                risk >= RISK_THRESHOLD and
                meta["seen_in_shelf_zone"] == 1 and
                in_exit and
                meta["frames_visible"] >= min_visible_frames and
                (frame_idx - meta["last_alert_frame"]) >= cooldown_frames
            ):
                print(f"[ALERT] f={frame_idx} id={tid} risk={risk:.2f} "
                      f"item={meta['item']} occ={meta['occluded']} speed={meta['speed']:.1f}")

                # Log to CSV
                with open(LOG_PATH, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([time.time(), frame_idx, tid, round(risk, 2),
                                meta["item"], meta["occluded"], round(meta["speed"], 1), VIDEO_PATH])

                meta["last_alert_frame"] = frame_idx
                alert_now = True

        # Save short clip around alert
        if alert_now and pending_after == 0:
            clip_frames = list(ring) + [frame.copy()]
            pending_after = post_frames
        elif pending_after > 0:
            clip_frames.append(frame.copy())
            pending_after -= 1
            if pending_after == 0 and clip_frames:
                save_clip(clip_frames, fps)
                clip_frames = []

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    # Flush pending clip
    if pending_after > 0 and clip_frames:
        save_clip(clip_frames, fps)

    cap.release()
    print("Done.")

if __name__ == "__main__":
    main()
