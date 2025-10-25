import argparse, json, subprocess
from pathlib import Path

def export(src, out_mp4, start_s, end_s):
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    # Use -ss before -i for fast seek; pad a tiny re-encode window for MP4 container safety
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.2f}", "-to", f"{end_s:.2f}",
        "-i", src,
        "-vf", "scale='min(854,iw)':'-2',fps=25,format=yuv420p",
        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
        "-an",
        out_mp4
    ]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="output/alert_events.json")
    ap.add_argument("--root_out", default="output/demos")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds before/after each event")
    args = ap.parse_args()

    events = json.loads(Path(args.events).read_text()).get("events", [])
    if not events:
        print("No events found in", args.events)
        return

    for i, e in enumerate(events, 1):
        src = e["src"]
        # If your CSV had absolute paths, keep them; otherwise they are relative to repo root
        start_s = e["start_offset_s"] - args.pad
        end_s   = e["end_offset_s"] + args.pad
        # Avoid negatives or inverted windows
        start_s = max(0.0, start_s)
        end_s   = max(start_s + 0.1, end_s)

        # Build filename: <stem>_track<T>_s<start>_e<end>_p<max>.mp4
        stem = Path(src).stem
        out = Path(args.root_out) / f"{stem}_track{e['tid']}_s{start_s:.2f}_e{end_s:.2f}_p{e['max_p']:.3f}.mp4"
        print(f"[{i}/{len(events)}] {src} -> {out.name}")
        try:
            export(src, str(out), start_s, end_s)
        except subprocess.CalledProcessError as ex:
            print("ffmpeg failed:", ex)

if __name__ == "__main__":
    main()
