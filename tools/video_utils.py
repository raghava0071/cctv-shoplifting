import cv2
import numpy as np

def read_frames_uniform(path, num_frames=12):
    """
    Read up to num_frames from a video, spaced as evenly as possible.
    Returns (frames, total_frames)
      - frames: list of BGR numpy arrays
      - total_frames: int
    If the video cannot be read, returns ([], 0)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], 0

    # choose indices
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            frames.append(frame)

    cap.release()
    return frames, total
