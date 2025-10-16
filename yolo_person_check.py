import os, sys
from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/test_store_clip.avi"
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: {VIDEO_PATH} not found.")
    sys.exit(1)

# Load YOLOv8 nano model (auto-downloads on first use)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Unable to open {VIDEO_PATH}.")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
print(f"Video open: frames={total_frames} fps={fps:.2f}")

person_name = "person"
checked = 0
total_person = 0

while checked < 120:  # check first ~12s if 10fps (or ~4s at 30fps)
    ret, frame = cap.read()
    if not ret:
        break
    checked += 1

    res = model(frame, conf=0.35, verbose=False)[0]
    names = model.model.names
    # count persons
    persons = 0
    if res.boxes is not None and res.boxes.cls is not None:
        for cls_idx in res.boxes.cls.tolist():
            cls_idx = int(cls_idx)
            if names.get(cls_idx, "") == person_name:
                persons += 1
    total_person += persons
    if checked % 10 == 0:
        print(f"Frame {checked:4d}: persons={persons} (running total={total_person})")

cap.release()
print(f"Done. Checked {checked} frames. Total person detections across checked frames = {total_person}. ✅")
