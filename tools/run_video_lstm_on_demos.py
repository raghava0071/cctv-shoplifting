#!/usr/bin/env python3
import os, sys, glob
import torch
import pandas as pd
import torch.nn as nn
from torchvision import models, transforms

# make project root importable
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)  # go up from tools/ → project root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from tools.video_utils import read_frames_uniform
except Exception as e:
    print("[ERR] cannot import read_frames_uniform:", e)
    print("[HINT] make sure tools/video_utils.py exists")
    raise

DEMOS_DIR = "output/demos"
CLEAN_MANIFEST = "output/demos_clean.csv"
OUT_CSV = "output/video_lstm_scores.csv"
MODEL_PATH = "models/video_lstm.pt"
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
        self.feat_dim = 512
        self.lstm = nn.LSTM(self.feat_dim, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        with torch.no_grad():
            feats = self.backbone(x)
        feats = feats.view(B, T, self.feat_dim)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        return self.fc(last)

def load_model(device):
    model = VideoLSTM()
    model = model.to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        # we saved {"model_state": ...}
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print("[INFO] loaded model from", MODEL_PATH)
        else:
            model.load_state_dict(ckpt)
            print("[INFO] loaded raw state_dict from", MODEL_PATH)
    else:
        print("[WARN] model not found, using random weights")
    model.eval()
    return model

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("[INFO] Using device:", device)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    # prefer clean manifest
    if os.path.exists(CLEAN_MANIFEST):
        df = pd.read_csv(CLEAN_MANIFEST)
        clip_paths = df["path"].tolist()
        print(f"[INFO] using manifest: {CLEAN_MANIFEST} ({len(clip_paths)} clips)")
    else:
        clip_paths = sorted(glob.glob(os.path.join(DEMOS_DIR, "*.mp4")))
        print(f"[INFO] using folder scan: {len(clip_paths)} clips")

    model = load_model(device)
    rows = []
    skipped = 0

    for i, path in enumerate(clip_paths, 1):
        frames, total = read_frames_uniform(path, NUM_FRAMES)
        if len(frames) == 0:
            print("[WARN] could not read frames from", path)
            skipped += 1
            continue

        if len(frames) < NUM_FRAMES:
            last = frames[-1]
            while len(frames) < NUM_FRAMES:
                frames.append(last)

        imgs = [tfm(f[..., ::-1]) for f in frames]  # BGR → RGB
        video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(video)
            prob = torch.softmax(logits, dim=1)[0,1].item()

        rows.append({
            "file": os.path.basename(path),
            "video_lstm_score": prob,
        })

        if i % 300 == 0:
            print(f"[INFO] processed {i}/{len(clip_paths)}")

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {len(rows)} rows to {OUT_CSV}, skipped {skipped}")
