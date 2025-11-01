#!/usr/bin/env python3
import os
import cv2
import math
import random
import pandas as pd
import numpy as np
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

LABEL_CSV = "output/labeled_clean.csv"
DEMOS_DIR = "output/demos"
MODEL_OUT = "models/video_lstm.pt"
NUM_FRAMES = 12   # how many frames per clip we sample
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 5
VAL_SPLIT = 0.2
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VideoClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, demos_dir: str, num_frames: int = 12, transform=None):
        self.df = df.reset_index(drop=True)
        self.demos_dir = demos_dir
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _read_frames_uniform(self, path: str, num_frames: int):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return []

        # pick indices uniformly
        idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["file"]
        label = int(row["label_int"])
        path = os.path.join(self.demos_dir, fname)

        frames = self._read_frames_uniform(path, self.num_frames)
        # if video was too short / failed, make dummy frames
        if len(frames) == 0:
            frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        # pad if fewer than num_frames
        if len(frames) < self.num_frames:
            last = frames[-1]
            while len(frames) < self.num_frames:
                frames.append(last)

        # apply transforms
        imgs = []
        for fr in frames:
            img = self.transform(fr)
            imgs.append(img)
        # (T, C, H, W)
        video = torch.stack(imgs, dim=0)
        return video, label, fname


class VideoLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2):
        super().__init__()
        # feature extractor: ResNet18 without FC
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(backbone.children())[:-1]  # remove FC
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.feature_dim = 512  # resnet18 output
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            feats = self.backbone(x)  # (B*T, 512, 1, 1)
        feats = feats.view(B, T, self.feature_dim)  # (B, T, 512)

        out, _ = self.lstm(feats)  # (B, T, H)
        last = out[:, -1, :]       # (B, H)
        logits = self.fc(last)
        return logits


def build_dataloaders():
    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"Missing {LABEL_CSV}. Run labeling step first.")

    df = pd.read_csv(LABEL_CSV)
    # keep only TRUE/FALSE
    df = df[df["label_int"].isin([0,1])].copy()
    df = df.reset_index(drop=True)

    # basic transform
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    dataset = VideoClipDataset(df, DEMOS_DIR, NUM_FRAMES, tfm)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, val_loader, len(dataset)


def train():
    set_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    train_loader, val_loader, total = build_dataloaders()
    print(f"[INFO] Dataset size: {total}")

    model = VideoLSTM(hidden_size=256, num_layers=1, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for videos, labels, _ in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for videos, labels, _ in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                logits = model(videos)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"[EPOCH {epoch}/{EPOCHS}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, MODEL_OUT)
            print(f"[INFO] Saved best model to {MODEL_OUT}")

    print("[DONE] best val acc:", best_val_acc)


if __name__ == "__main__":
    train()
