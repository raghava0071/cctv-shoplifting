import os, time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CCTV Alerts Review", layout="wide")
st.title("🛒 CCTV Shoplifting Prototype — Alerts Review")

alerts_dir = "output/alerts"
log_path = "output/alert_log.csv"

left, right = st.columns([1,2], gap="large")

with left:
    st.subheader("Alert Log")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = df.sort_values("ts_unix", ascending=False)
        df["when"] = pd.to_datetime(df["ts_unix"], unit="s")
        df = df[["when","frame","track_id","risk","item","occluded","speed","video"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No alert_log.csv yet. Run risk_alerts.py first.")

with right:
    st.subheader("Alert Clips")
    if os.path.exists(alerts_dir):
        files = sorted([f for f in os.listdir(alerts_dir) if f.endswith(".mp4")], reverse=True)
        if not files:
            st.info("No alert clips found yet.")
        else:
            for f in files[:12]:
                st.markdown(f"**{f}**")
                st.video(os.path.join(alerts_dir, f))
                st.divider()
    else:
        st.info("alerts directory not found.")
