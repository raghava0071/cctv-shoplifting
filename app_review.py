import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="CCTV Shoplifting Review", layout="wide")

st.title("CCTV Shoplifting – Human-in-the-Loop Review")

FINAL_CSV = "output/final_scores.csv"
DEMOS_DIR = "output/demos"

if not os.path.exists(FINAL_CSV):
    st.error("output/final_scores.csv not found. Run tools/build_final_ensemble.py first.")
    st.stop()

df = pd.read_csv(FINAL_CSV)

st.sidebar.header("Filters")

label_filter = st.sidebar.multiselect(
    "Label (human)",
    options=sorted(df["label"].dropna().unique().tolist()),
    default=[],
)

min_score = st.sidebar.slider(
    "Minimum final score",
    min_value=0.0,
    max_value=float(df["final_score"].max()),
    value=0.5,
    step=0.01,
)

show_n = st.sidebar.slider("Show top N", 10, 200, 50, 10)

# apply filters
f = df.copy()
if label_filter:
    f = f[f["label"].isin(label_filter)]
f = f[f["final_score"] >= min_score]
f = f.sort_values("final_score", ascending=False).head(show_n)

st.write(f"Showing {len(f)} clips")

for _, row in f.iterrows():
    clip_rel = row["clip_path"]
    clip_abs = os.path.join(DEMOS_DIR, clip_rel)
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader(os.path.basename(clip_rel))
        st.write(f"label={row.get('label','?')}  | final_score={row['final_score']:.3f}")
        st.write(f"yolo={row.get('yolo_prob',0):.3f}  frame={row.get('frame_cnn_score',0):.3f}  video={row.get('video_lstm_score',0):.3f}")
        if os.path.exists(clip_abs):
            st.video(clip_abs)
        else:
            st.warning("video file missing (maybe deleted as broken)")
    with cols[1]:
        st.json({
            "file": row.get("file", clip_rel),
            "final_score": float(row["final_score"]),
            "yolo_prob": float(row.get("yolo_prob", 0)),
            "frame_cnn_score": float(row.get("frame_cnn_score", 0)),
            "video_lstm_score": float(row.get("video_lstm_score", 0)),
            "label": row.get("label", "")
        })
