from pathlib import Path
import numpy as np
import torch
from torch import nn

class LSTMBinary(nn.Module):
    def __init__(self, input_dim=51, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=1, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        hcat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(hcat)

class BehaviorScorer:
    """
    Loads runs/lstm_skeleton_best.pt and scores one pose track (np.ndarray T x 17 x 3)
    Returns probability of 'shoplifting' (float 0..1).
    """
    def __init__(self, ckpt_path="runs/lstm_skeleton_best.pt", device=None, max_len=160):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_len = max_len
        self.model = LSTMBinary(input_dim=17*3).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _normalize(arr):
        # arr: (T,17,3) with (x,y,conf). Center at hips, scale by spread.
        xy = arr[..., :2]
        c  = arr[..., 2:3]
        hip_idx = [11, 12] if arr.shape[1] >= 13 else [0]
        hip = xy[:, hip_idx, :].mean(axis=1, keepdims=True)
        xy = xy - hip
        spread = np.maximum(1e-3, np.linalg.norm(xy.reshape(xy.shape[0], -1), axis=1, keepdims=True).mean())
        xy = xy / spread
        return np.concatenate([xy, c], axis=-1)

    def _pad_or_trim(self, arr):
        T = arr.shape[0]
        if T == self.max_len: return arr
        if T > self.max_len:
            s = (T - self.max_len)//2
            return arr[s:s+self.max_len]
        reps = (self.max_len + T - 1)//T
        return (np.concatenate([arr]*reps, axis=0))[:self.max_len]

    def score_track(self, pose_track: np.ndarray) -> float:
        """
        pose_track: numpy array (T,17,3) as saved by our extractor.
        Returns: probability of 'shoplifting' (float 0..1)
        """
        arr = self._normalize(pose_track)
        arr = self._pad_or_trim(arr)
        x = arr.reshape(arr.shape[0], -1).astype(np.float32)  # (T,51)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device) # (1,T,51)
        with torch.no_grad():
            logits = self.model(xt)
            prob = torch.softmax(logits, dim=-1)[0,1].item()
        return float(prob)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help="path to pose_out/.../track_XX.npy")
    ap.add_argument("--ckpt", default="runs/lstm_skeleton_best.pt")
    args = ap.parse_args()

    scorer = BehaviorScorer(ckpt_path=args.ckpt)
    import numpy as np
    arr = np.load(args.track)
    p = scorer.score_track(arr)
    print(f"Shoplifting probability: {p:.3f}")
