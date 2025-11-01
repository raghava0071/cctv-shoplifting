#!/usr/bin/env python3
import os
import cv2
import random
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

DATA_CSV = "output/labeled_clean.csv"
DEMOS_DIR = "output/demos"
MODEL_OUT = "models/second_stage_frame_cnn.pt"
BATCH_SIZE = 16
EPOCHS = 5
VAL_SPLIT = 0.2
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

def read_middle_frame(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        cap.release()
        return None
    mid = length // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

class CCTVFrameDataset(Dataset):
    def __init__(self, csv_path, demos_dir, tfm=None):
        self.df = pd.read_csv(csv_path)
        # keep only rows that we can actually load as videos
        records = []
        for _, row in self.df.iterrows():
            fname = row["file"]
            label = int(row["label_int"])
            vpath = os.path.join(demos_dir, fname)
            if os.path.exists(vpath):
                records.append((vpath, label))
        self.data = records
        self.tfm = tfm
        print(f"[INFO] Dataset has {len(self.data)} video clips")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vpath, label = self.data[idx]
        frame = read_middle_frame(vpath)
        if frame is None:
            # return a black image if failed
            import numpy as np
            frame = np.zeros((224,224,3), dtype="uint8")
        if self.tfm:
            frame = self.tfm(frame)
        return frame, label

def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found")
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    full_ds = CCTVFrameDataset(DATA_CSV, DEMOS_DIR, tfm=tfm)
    n_total = len(full_ds)
    if n_total == 0:
        print("[ERR] No usable videos found under output/demos")
        return
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 2)
    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # eval
        model.eval()
        val_correct = 0
        val_total = 0
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                for p, y in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    if p == 1 and y == 1: tp += 1
                    elif p == 1 and y == 0: fp += 1
                    elif p == 0 and y == 0: tn += 1
                    elif p == 0 and y == 1: fn += 1
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0

        print(f"[EPOCH {epoch}/{EPOCHS}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_acc={val_acc:.3f} prec={prec:.3f} rec={rec:.3f} "
              f"TP={tp} FP={fp} TN={tn} FN={fn}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, MODEL_OUT)
            print(f"[INFO] Saved best model to {MODEL_OUT}")

    print("[DONE] best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
