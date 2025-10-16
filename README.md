# CCTV Shoplifting / Surveillance Risk Prototype

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

