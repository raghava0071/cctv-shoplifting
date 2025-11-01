#!/usr/bin/env python3
import os
import glob
import csv
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

DEMOS_DIR = "output/demos"
MODEL_PATH = "models/second_stage_frame_cnn.pt"
OUT_CSV = "output/frame_cnn_scores.csv"

def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=device)

    # handle both styles: plain state_dict or {"model": ..., ...}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    # detect how many outputs the trained model had
    if "fc.weight" not in state:
        raise RuntimeError("Checkpoint does not contain fc.weight — did you save correctly?")
    num_classes = state["fc.weight"].shape[0]   # 1 or 2

    # build model with SAME num_classes
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, num_classes

def extract_center_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    center_idx = max(total // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, center_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def main():
    os.makedirs("output", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model, num_classes = load_model(device)
    print(f"[INFO] Loaded model with {num_classes} output(s)")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    clips = sorted(glob.glob(os.path.join(DEMOS_DIR, "*.mp4")))
    if not clips:
        print(f"[WARN] No clips found in {DEMOS_DIR}")
        return

    rows = []
    for i, clip in enumerate(clips, 1):
        frame = extract_center_frame(clip)
        if frame is None:
            print(f"[WARN] could not read frame from {clip}")
            continue
        x = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)  # shape: [1, C]
            if num_classes == 2:
                probs = torch.softmax(logits, dim=1)
                # assume class 1 = "shoplifting"
                prob = probs[0, 1].item()
            else:
                # 1-output sigmoid
                prob = torch.sigmoid(logits[0, 0]).item()

        rows.append(dict(file=os.path.basename(clip),
                         path=clip,
                         prob=prob))

        if i % 50 == 0:
            print(f"[INFO] processed {i}/{len(clips)}")

    # write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "path", "prob"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[DONE] wrote {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
