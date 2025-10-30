from collections import defaultdict, deque
import numpy as np
from typing import Optional
from models.behavior_scorer import BehaviorScorer

class BehaviorRuntime:
    def __init__(self, ckpt_path="runs/lstm_skeleton_best.pt", threshold=0.51, winlen=120, minlen=32):
        self.scorer = BehaviorScorer(ckpt_path=ckpt_path, threshold=threshold, max_len=winlen)
        self.winlen = winlen
        self.minlen = minlen
        self.buffers = defaultdict(lambda: deque(maxlen=self.winlen))

    def update(self, track_id: int, keypoints_17x3_np: np.ndarray):
        """Append (17,3) keypoints for this track for the current frame."""
        # Expected shape: (17,3) for the current frame (x,y,conf)
        if keypoints_17x3_np is None or keypoints_17x3_np.shape != (17,3):
            return
        self.buffers[int(track_id)].append(keypoints_17x3_np)

    def score(self, track_id: int) -> Optional[float]:
        """Return shoplifting probability for this track if buffer is long enough, else None."""
        buf = self.buffers.get(int(track_id))
        if not buf or len(buf) < self.minlen:
            return None
        seq = np.stack(list(buf), axis=0)  # (T,17,3)
        return float(self.scorer.score_track(seq))
