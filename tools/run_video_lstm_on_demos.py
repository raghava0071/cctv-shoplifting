#!/usr/bin/env python3
import os
import cv2
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

DEMOS_DIR = "output/demos"
MODEL_PATH = "models/video_lstm.pt"
OUT_CSV = "output/video_lstm_scores.csv"
NUM_FRAMES = 12
IMG_SIZE = 224


class VideoLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.feature_dim = 512
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
        x = x.view(B*T, C, H, W)
        with torch.no_grad():
            feats = self.backbone(x)
        feats = feats.view(B, T, self.feature_dim)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def read_frames_uniform(path, num_frames=12):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, n-1, num_frames, dtype=int)
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


def load_model(device):
    model = VideoLSTM()
    model = model.to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print("[INFO] Loaded model from", MODEL_PATH)
    else:
        print("[WARN] model not found, using random weights")
    model.eval()
    return model


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    model = load_model(device)

    clips = sorted(glob.glob(os.path.join(DEMOS_DIR, "*.mp4")))
    rows = []

    for i, path in enumerate(clips, 1):
        frames = read_frames_uniform(path, NUM_FRAMES)
        if len(frames) == 0:
            print("[WARN] could not read frames from", path)
            continue

        if len(frames) < NUM_FRAMES:
            last = frames[-1]
            while len(frames) < NUM_FRAMES:
                frames.append(last)

        imgs = [tfm(fr) for fr in frames]
        video = torch.stack(imgs, dim=0)  # (T, C, H, W)
        video = video.unsqueeze(0).to(device)  # (1, T, C, H, W)

        with torch.no_grad():
            logits = model(video)
            probs = torch.softmax(logits, dim=1)
            shoplift_score = float(probs[0,1].item())

        rows.append({
            "file": os.path.basename(path),
            "video_lstm_score": shoplift_score,
        })

        if i % 100 == 0:
            print(f"[INFO] processed {i}/{len(clips)} clips")

    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
