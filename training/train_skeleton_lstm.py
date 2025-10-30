import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

LABEL_MAP = {"normal": 0, "shoplifting": 1}

def read_rows(labels_csv):
    return list(csv.DictReader(open(labels_csv)))

def stratified_split(rows, split=0.85, seed=42):
    by_cls = {"normal": [], "shoplifting": []}
    for r in rows:
        by_cls[r["label"].strip().lower()].append(r)
    rnd = random.Random(seed)
    tr, va = [], []
    for cls, lst in by_cls.items():
        rnd.shuffle(lst)
        ntr = max(1, int(len(lst)*split)) if len(lst)>1 else len(lst)
        tr += lst[:ntr]; va += lst[ntr:]
    # If val got 0 of a class due to tiny counts, move one from train
    need = [c for c in ["normal","shoplifting"] if not any(r["label"]==c for r in va) and any(r["label"]==c for r in tr)]
    for c in need:
        for i,r in enumerate(tr):
            if r["label"]==c:
                va.append(tr.pop(i)); break
    return tr, va

class PoseTrackDataset(Dataset):
    def __init__(self, rows, max_len=160, min_len=48, pad_mode="repeat"):
        self.items = [(r["path"], LABEL_MAP[r["label"].strip().lower()]) for r in rows]
        self.max_len = max_len
        self.min_len = min_len
        self.pad_mode = pad_mode

    def __len__(self): return len(self.items)

    def _normalize(self, arr):
        xy = arr[..., :2]; c = arr[..., 2:3]
        hip_idx = [11,12] if arr.shape[1]>=13 else [0]
        hip = xy[:, hip_idx, :].mean(axis=1, keepdims=True)
        xy = xy - hip
        spread = np.maximum(1e-3, np.linalg.norm(xy.reshape(xy.shape[0], -1), axis=1, keepdims=True).mean())
        xy = xy / spread
        return np.concatenate([xy, c], axis=-1)

    def _pad_or_trim(self, arr):
        T = arr.shape[0]
        if T >= self.max_len:
            s = (T - self.max_len)//2
            return arr[s:s+self.max_len]
        reps = (self.max_len + T - 1)//T
        return np.concatenate([arr]*reps, axis=0)[:self.max_len]

    def __getitem__(self, idx):
        p, y = self.items[idx]
        arr = np.load(p)
        if arr.shape[0] < self.min_len:
            return self.__getitem__((idx+1)%len(self.items))
        arr = self._normalize(arr)
        arr = self._pad_or_trim(arr)
        x = arr.reshape(arr.shape[0], -1).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.softmax(logits, dim=-1)[torch.arange(logits.size(0)), target]
        loss = ((1-pt)**self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()

class LSTMBinary(nn.Module):
    def __init__(self, input_dim, hidden=192):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=2, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden*2, 256), nn.ReLU(), nn.Linear(256, 2))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        hcat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(hcat)

def class_weights(rows):
    from collections import Counter
    c = Counter(r["label"].strip().lower() for r in rows)
    n0, n1 = c.get("normal",0), c.get("shoplifting",0)
    total = max(1, n0+n1)
    w0 = total/max(1,n0); w1 = total/max(1,n1)
    s = w0+w1
    import torch
    return torch.tensor([w0/s, w1/s], dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--min_len", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    Path("runs").mkdir(exist_ok=True)
    rows = read_rows(args.labels)
    tr_rows, va_rows = stratified_split(rows, split=0.85, seed=42)
    print("Split sizes -> train:", len(tr_rows), "val:", len(va_rows))

    train_ds = PoseTrackDataset(tr_rows, max_len=args.max_len, min_len=args.min_len)
    val_ds   = PoseTrackDataset(va_rows, max_len=args.max_len, min_len=args.min_len)
    if len(train_ds)==0 or len(val_ds)==0: raise SystemExit("Empty train/val after filtering. Lower --min_len or add data.")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)
    model = LSTMBinary(input_dim=17*3).to(device)

    alpha = class_weights(rows).to(device)
    crit  = FocalLoss(alpha=alpha, gamma=2.0)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_f1, patience, bad = 0.0, 6, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        tl, tp, pp, rp, n = 0.0, 0, 0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            tl += loss.item() * x.size(0)
            pred = (torch.softmax(logits, dim=-1)[:,1] > 0.5).long()
            tp += ((pred==1) & (y==1)).sum().item()
            pp += (pred==1).sum().item()
            rp += (y==1).sum().item()
            n  += x.size(0)
        tl /= max(1,n)
        precision = tp/max(1,pp); recall = tp/max(1,rp); f1 = 2*precision*recall/max(1e-8, precision+recall)

        model.eval()
        vtp,vpp,vrp,vn = 0,0,0,0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pr = torch.softmax(model(x), dim=-1)[:,1]
                pred = (pr > 0.5).long()
                vtp += ((pred==1) & (y==1)).sum().item()
                vpp += (pred==1).sum().item()
                vrp += (y==1).sum().item()
                vn  += x.size(0)
        vprec = vtp/max(1,vpp); vrec = vtp/max(1,vrp); vf1 = 2*vprec*vrec/max(1e-8, vprec+vrec)

        sched.step()
        print(f"Epoch {epoch:02d} | train_loss {tl:.4f} | train_P {precision:.3f} R {recall:.3f} F1 {f1:.3f} | val_P {vprec:.3f} R {vrec:.3f} F1 {vf1:.3f}")
        if vf1 > best_f1:
            best_f1 = vf1; bad = 0
            torch.save(model.state_dict(), "runs/lstm_skeleton_best.pt")
            print("  [*] Saved runs/lstm_skeleton_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

if __name__ == "__main__":
    main()
