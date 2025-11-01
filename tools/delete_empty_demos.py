#!/usr/bin/env python3
import os, glob, cv2, shutil

DEMOS = "output/demos"
TRASH = "output/demos_trash_empty"
os.makedirs(TRASH, exist_ok=True)

moved = 0
for path in glob.glob(os.path.join(DEMOS, "*.mp4")):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        shutil.move(path, os.path.join(TRASH, os.path.basename(path)))
        print("[DEL] bad (not opened):", path)
        moved += 1
        continue
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames <= 0:
        shutil.move(path, os.path.join(TRASH, os.path.basename(path)))
        print("[DEL] bad (0 frames):", path)
        moved += 1

print(f"[DONE] moved {moved} files to {TRASH}")
