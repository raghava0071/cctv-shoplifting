# CCTV Shoplifting / Surveillance Risk Prototype
## Human-in-the-Loop Shoplifting Evaluation

After generating candidate alert clips from the DCSASS shoplifting dataset, I built a small review loop to see how well the model’s score matched real shoplifting behavior.

**Goal:** find out if a single fixed threshold (0.55) is good enough, and what kinds of scenes the model is actually confused by.

### Steps

1. **Collect alerts**  
   All model detections were written to `output/alert_log.csv` with fields  
   `ts, track_id, frames, shoplift_prob, fired, source, clip_path`.

2. **Generate demo clips**  
   Using `tools/export_event_clips.py` I exported short MP4s into `output/demos/` for manual review.

3. **Manual labeling**  
   I opened only the in-store / “Shoplifting*.mp4” clips and labeled **896** of them as:
   - `TRUE` – real shoplifting / suspicious behavior
   - `FALSE` – normal motion, walking, background people
   - `UNCERTAIN` – low visibility / ambiguous

   Labels were saved to: `output/labels.csv`

4. **Clean labels**  
   I converted them to a trainable CSV:

   ```bash
   python - <<'PY'
   import re, pandas as pd, numpy as np, os
   df = pd.read_csv("output/labels.csv")

   def extract_prob(name):
       import re
       m = re.search(r"_p([0-9]+\.[0-9]+)(?=\.mp4|_)", str(name))
       return float(m.group(1)) if m else np.nan

   df["prob"] = df["file"].apply(extract_prob)
   df["label_int"] = df["label"].str.upper().map({"TRUE":1, "FALSE":0})

   clean = df.dropna(subset=["prob","label_int"]).copy()
   os.makedirs("output", exist_ok=True)
   clean.to_csv("output/labeled_clean.csv", index=False)
   print("Wrote output/labeled_clean.csv with", len(clean), "rows")
   PY


> Real-time demo that detects people, tracks movement, and flags **suspicious shelf→exit behavior** from CCTV video. Saves short alert clips and a CSV event log for human review.

![status](https://img.shields.io/badge/status-prototype-blue)
![python](https://img.shields.io/badge/Python-3.10%2B-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

## ✨ Features
- YOLOv8 person & carry-item detection (bags, bottle, phone)
- DeepSORT multi-object tracking with stable IDs
- Heuristic risk score (zones + item overlap + occlusion + speed)
- **Alerting:** 2s video clips + `output/alert_log.csv`
- Minimal **Streamlit** reviewer (`app_review.py`)

## 📦 Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

