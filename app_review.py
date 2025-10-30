import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="CCTV Alerts Review", layout="wide")
st.title("CCTV Alerts Review")

csv_path = Path("output/alert_log.csv")
if not csv_path.exists():
    st.warning("No alert_log.csv found yet.")
    st.stop()

df = pd.read_csv(csv_path)

# Expected columns from risk_alerts.py
expected = ["ts","track_id","frames","shoplift_prob","fired","source","clip_path"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.dataframe(df.head())
    st.stop()

# Filters
c1, c2, c3 = st.columns(3)
with c1:
    min_prob = st.slider("Min probability", 0.0, 1.0, 0.50, 0.01)
with c2:
    only_fired = st.checkbox("Only fired events", value=True)
with c3:
    min_frames = st.number_input("Min frames", min_value=0, value=16, step=1)

fdf = df.copy()
fdf = fdf[fdf["shoplift_prob"] >= min_prob]
if only_fired:
    fdf = fdf[fdf["fired"] == 1]
fdf = fdf[fdf["frames"] >= min_frames]

st.subheader("Alerts")
st.dataframe(fdf.sort_values("ts").tail(500), use_container_width=True)

st.subheader("Sources")
st.write(fdf["source"].dropna().unique())
