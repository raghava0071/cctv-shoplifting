import os
import glob
import time
import cv2
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CCTV Alerts Review", layout="wide")
st.title("🛒 CCTV Shoplifting Prototype — Alerts Review")

ALERTS_DIR = "output/alerts"
LOG_PATH = "output/alert_log.csv"

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Alert Log")
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        if "ts_unix" in df.columns:
            df = df.sort_values("ts_unix", ascending=False)
            df["when"] = pd.to_datetime(df["ts_unix"], unit="s")
            keep_cols = [c for c in ["when", "frame", "track_id", "risk", "item", "occluded", "speed", "video"] if c in df.columns]
            df = df[keep_cols]
        # Use a fixed pixel width for maximum compatibility
        st.dataframe(df, width=1000)
    else:
        st.info("No alert_log.csv yet. Run risk_alerts.py first.")

with right:
    st.subheader("Alert Clips")
    if not os.path.exists(ALERTS_DIR):
        st.info("alerts directory not found.")
    else:
        # Prefer *_web.mp4 (web-optimized) over raw .mp4
        all_mp4 = sorted(glob.glob(os.path.join(ALERTS_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
        preferred = []
        raw_seen = set()
        for f in all_mp4:
            if f.endswith("_web.mp4"):
                raw = f[:-8] + ".mp4"
                raw_seen.add(raw)
                preferred.append(f)
        for f in all_mp4:
            if not f.endswith("_web.mp4") and f not in raw_seen:
                preferred.append(f)

        files = preferred
        if not files:
            st.info("No alert clips found yet.")
        else:
            for f in files[:12]:
                fname = os.path.basename(f)
                try:
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                except OSError:
                    size_mb = 0.0

                # Try to read fps/frames for a duration estimate
                dur_txt = ""
                try:
                    cap = cv2.VideoCapture(f)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    duration = (frames / fps) if fps > 0 else 0.0
                    dur_txt = f" — {size_mb:.2f} MB | {duration:.2f}s @ {fps:.1f}fps, {frames} frames"
                except Exception:
                    dur_txt = f" — {size_mb:.2f} MB"

                st.markdown(f"**{fname}**{dur_txt}")

                # Serve bytes (more reliable than passing a filesystem path)
                try:
                    with open(f, "rb") as fh:
                        data = fh.read()
                    st.video(data, format="video/mp4")
                except Exception as e:
                    st.warning(f"Could not load video: {e}")

                st.divider()
