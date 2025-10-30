#!/usr/bin/env python3
import os
import glob
import pandas as pd
import cv2

ALERT_LOG = "output/alert_log.csv"
DEMOS_DIR = "output/demos"
ALERTS_DIR = "output/alerts"
MIN_DEMO_BYTES = 1024 * 30  # ignore empty ~261B clips


def best_demo_for_source(src: str):
    """Find demo clip matching the alert source."""
    stem = os.path.splitext(os.path.basename(src))[0]
    candidates = sorted(glob.glob(os.path.join(DEMOS_DIR, f"{stem}*.mp4")))
    candidates = [c for c in candidates if os.path.exists(c) and os.path.getsize(c) >= MIN_DEMO_BYTES]
    return candidates[0] if candidates else None


def pick_top_sources(df: pd.DataFrame, k: int = 10):
    """Return top-k sources by max shoplift probability."""
    df_use = df[df["fired"] == 1] if "fired" in df.columns else df
    if len(df_use) == 0:
        df_use = df
    agg = df_use.groupby("source", dropna=True)["shoplift_prob"].max().sort_values(ascending=False)
    pairs = [(s, float(p)) for s, p in agg.head(k).items()]
    kept = []
    for s, p in pairs:
        if s and isinstance(s, str) and os.path.exists(s):
            kept.append((s, p))
    return kept


def looks_like_store(path: str) -> bool:
    """Heuristic: only include in-store/shop videos."""
    name = os.path.basename(path).lower()
    return any(k in name for k in ["shop", "store", "aisle", "retail", "checkout", "market"])


def play_video(path: str, overlay: str = "", fps_cap: float = 0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = int(1000 / (fps_cap if fps_cap > 0 else fps))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if overlay:
            cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, os.path.basename(path), (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Shoplifting Review (n=next, p=prev, q=quit)", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)
        elif key == ord("n"):
            cap.release()
            cv2.destroyAllWindows()
            return "next"
        elif key == ord("p"):
            cap.release()
            cv2.destroyAllWindows()
            return "prev"
    cap.release()
    cv2.destroyAllWindows()
    return None


def main():
    if not os.path.exists(ALERT_LOG):
        print(f"[ERR] Missing {ALERT_LOG}")
        return
    df = pd.read_csv(ALERT_LOG)
    df["source"] = df["source"].astype(str)

    # pick top 200 sources and filter to store-looking names
    top_all = pick_top_sources(df, k=200)
    top = [(s, p) for (s, p) in top_all if looks_like_store(s)]
    if not top:
        top = top_all[:20]

    playlist = []
    for src, prob in top:
        demo = best_demo_for_source(src)
        if demo:
            playlist.append((demo, prob, "(demo)"))
        else:
            playlist.append((src, prob, "(source)"))

    print("Playlist (highest score first):")
    for i, (path, prob, tag) in enumerate(playlist, 1):
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"{i:2d}. {tag} {prob:.3f} {path} [{size} bytes]")

    i = 0
    while 0 <= i < len(playlist):
        path, prob, tag = playlist[i]
        overlay = f"{tag}  score={prob:.3f}"
        nav = play_video(path, overlay=overlay, fps_cap=25)
        if nav == "next":
            i = min(i + 1, len(playlist) - 1)
        elif nav == "prev":
            i = max(i - 1, 0)
        else:
            i += 1


if __name__ == "__main__":
    main()

