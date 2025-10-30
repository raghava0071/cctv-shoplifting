import argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

class LSTMBinary(nn.Module):
    def __init__(self, input_dim=51, hidden=192):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=2, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden*2,256), nn.ReLU(), nn.Linear(256,2))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        hcat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(hcat)

def normalize(arr):
    xy = arr[..., :2]; c  = arr[..., 2:3]
    hip_idx = [11, 12] if arr.shape[1] >= 13 else [0]
    hip = xy[:, hip_idx, :].mean(axis=1, keepdims=True)
    xy = (xy - hip)
    spread = np.maximum(1e-3, np.linalg.norm(xy.reshape(xy.shape[0], -1), axis=1, keepdims=True).mean())
    xy = xy / spread
    return np.concatenate([xy, c], axis=-1)

def pad_or_trim(arr, L=160):
    T = arr.shape[0]
    if T >= L:
        s = (T - L)//2
        return arr[s:s+L]
    reps = (L + T - 1)//T
    return np.concatenate([arr]*reps, axis=0)[:L]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpt", default="runs/lstm_skeleton_best.pt")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LSTMBinary().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    xs, ys, paths = [], [], []
    for r in csv.DictReader(open(args.labels)):
        p = r["path"]; y = 1 if r["label"].strip().lower()=="shoplifting" else 0
        arr = np.load(p); arr = normalize(arr); arr = pad_or_trim(arr, 160)
        x = torch.from_numpy(arr.reshape(arr.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pr = torch.softmax(model(x), dim=-1)[0,1].item()
        xs.append(pr); ys.append(y); paths.append(p)

    try:
        roc = roc_auc_score(ys, xs)
        ap  = average_precision_score(ys, xs)
    except ValueError:
        roc, ap = float('nan'), float('nan')

    ps, rs, ts = precision_recall_curve(ys, xs)
    f1s = 2*ps*rs/(ps+rs+1e-9)
    best_idx = int(np.nanargmax(f1s))
    best_t = ts[best_idx-1] if best_idx>0 and best_idx<=len(ts) else 0.5
    pred = [1 if s>=best_t else 0 for s in xs]
    f1  = f1_score(ys, pred)

    print(f"ROC-AUC: {roc:.3f}  PR-AUC: {ap:.3f}  Best-F1: {f1:.3f} @ thresh={best_t:.3f}")

    out = Path("runs/predictions.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path","label","score","pred_at_bestF1"])
        for p,y,s,pp in zip(paths,ys,xs,pred):
            w.writerow([p, y, f"{s:.6f}", pp])
    print("Wrote", out)

if __name__ == "__main__":
    main()
