import cv2, sys, os

VIDEO_PATH = "data/test_store_clip.avi"

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: {VIDEO_PATH} not found. Make sure the file exists.")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Unable to open {VIDEO_PATH}.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
print(f"Opened {VIDEO_PATH}")
print(f"FPS: {fps:.2f}  Frames: {total}  Resolution: {w}x{h}")

# read a few frames to confirm decode works
ok_frames = 0
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    ok_frames += 1

cap.release()
print(f"Read {ok_frames} frames successfully. ✅")
