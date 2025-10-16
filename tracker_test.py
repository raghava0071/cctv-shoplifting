import os, cv2, sys
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "data/test_store_clip.avi"
os.makedirs("output", exist_ok=True)

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = "output/tracked_people.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    results = model(frame, conf=0.35, verbose=False)[0]

    detections = []
    for box, cls_idx, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        cls_name = model.model.names[int(cls_idx)]
        if cls_name == "person":  # track only people
            x1,y1,x2,y2 = map(int, box.tolist())
            w, h = x2 - x1, y2 - y1
            detections.append(([x1,y1,w,h], conf, cls_name))

    tracks = tracker.update_tracks(detections, frame=frame)
    active_tracks = 0
    for t in tracks:
        if not t.is_confirmed():
            continue
        active_tracks += 1
        tid = t.track_id
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx:4d}: active_tracks={active_tracks}")
    writer.write(frame)

cap.release()
writer.release()
print(f"✅ Done. Saved output video: {out_path}")
