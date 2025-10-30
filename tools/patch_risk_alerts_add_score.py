# Paste these snippets into your risk_alerts.py at the appropriate places.
# 1) Top imports
# from collections import deque, defaultdict
# from models.behavior_scorer import BehaviorScorer
#
# 2) Global init (once)
# pose_buffers = defaultdict(lambda: deque(maxlen=120))
# behavior_scorer = BehaviorScorer(ckpt_path="runs/lstm_skeleton_best.pt", threshold=0.51, max_len=120)
#
# 3) Per-frame, after you have track_id and keypoints (keypoints as (17,3)):
# pose_buffers[track_id].append(keypoints)  # keypoints = numpy (17,3)
#
# 4) When you decide to create an alert clip for a given track:
# seq = np.stack(list(pose_buffers[track_id]), axis=0)
# prob = behavior_scorer.score_track(seq)
# # use prob in your event metadata:
# event_meta = {
#    "track_id": track_id,
#    "shoplift_prob": float(prob),
#    ...
# }
# # then save event_meta to your CSV or DB along with clip path
#
# 5) If you already save alert_log.csv as CSV rows, add a column "shoplift_prob" and write the value.
