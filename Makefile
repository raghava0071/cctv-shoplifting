VENV=venv/bin/activate

extract:
	. $(VENV); python tools/extract_pose_tracks.py --glob "output/alerts/*.mp4" --out pose_out

labels:
	. $(VENV); python tools/label_review.py

train:
	. $(VENV); python training/train_skeleton_lstm.py --labels labels.csv --epochs 30 --batch 48 --max_len 160 --min_len 32

eval:
	. $(VENV); python training/eval_skeleton_lstm.py --labels labels.csv

realtime:
	. $(VENV); PYTHONPATH=. python -m runtime.realtime_behavior --source "output/alerts/alert_1760582836220.mp4" --threshold 0.51 --minlen 8 --display 0

alerts:
	. $(VENV); python risk_alerts.py --source "output/alerts/alert_1760582836220.mp4" --threshold 0.51 --display 0
