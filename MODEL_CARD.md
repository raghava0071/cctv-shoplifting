# CCTV Shoplifting – Model Card

## 1. Overview
This repository builds a human-in-the-loop CCTV shoplifting detector. The final score is an ensemble of:
1. YOLOv8-style risk/alert score (from clip filename, e.g. `_p0.553`)
2. Frame-level CNN (ResNet18, trained on 714 human-labeled clips)
3. Video-level LSTM (runs on decoded frames, robust to motion)

## 2. Intended Use
- Retail loss-prevention prototype
- Triage of many short alert clips
- Ranking clips for human review in a Streamlit UI

**Not** for: fully automated police / security response.

## 3. Data
- Source: DCSASS Shoplifting subset (local, not pushed to GitHub)
- 896 clips manually labeled by human
- 714 clips usable after cleaning:
  - TRUE: ~145
  - FALSE: ~569
  - UNCERTAIN: dropped
- Broken / empty MP4s moved to `output/demos_trash_empty/`

## 4. Models
- **Stage 1 (alert):** YOLOv8 + tracking → short MP4s → score in filename
- **Stage 2 (frame):** `training/train_frame_cnn.py` (ResNet18, mps)
  - best val acc ≈ 0.96
- **Stage 3 (video):** `training/train_video_lstm.py`
  - trained on cleaned demos
- **Ensemble:** `tools/build_final_ensemble.py`
  - takes `manifests/clean_demos.csv`, joins CNN + LSTM scores

## 5. Evaluation
After merging scores with human labels:
- high recall at low thresholds (0.53–0.55)
- precision improves when frame-CNN + video-LSTM agree
- disagreements exported to: `output/frame_cnn_disagreements/`

## 6. Limitations
- Some alerts were empty MP4s from ffmpeg → filtered out
- Dataset is single-domain (indoor, fixed cam)
- Scores in filenames can be noisy → we re-parse using regex
- Real-time deployment would need ONNX / TensorRT

## 7. Reproduce
```bash
# 1) run detector → demos
python starter_pipeline.py

# 2) clean demos
python tools/find_bad_demos.py
python tools/delete_empty_demos.py

# 3) run DL heads
python tools/run_frame_cnn_on_demos.py
python tools/run_video_lstm_on_demos.py

# 4) build ensemble
python tools/build_clean_manifest.py --demos-dir output/demos --bad-list output/bad_demos/list.txt --out manifests/clean_demos.csv
python tools/build_final_ensemble.py --clean-manifest manifests/clean_demos.csv --frame-csv output/frame_cnn_scores.csv --video-csv output/video_lstm_scores.csv --out manifests/final_ensemble.csv

Author

Raghavendra Karanam
