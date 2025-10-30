#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
import math, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--events", default="output/alert_events.json")
parser.add_argument("--root_out", default="output/demos")
parser.add_argument("--pad", type=float, default=0.5)
parser.add_argument("--min_clip", type=float, default=0.8)  # min duration
args = parser.parse_args()

with open(args.events) as f:
    data = json.load(f)
events = data.get("events", [])
os.makedirs(args.root_out, exist_ok=True)
for i,e in enumerate(events):
    src = Path(e["src"])
    if not src.exists():
        print("Source missing:", src); continue
    start = max(0, e["start_offset_s"] - args.pad)
    end = e["end_offset_s"] + args.pad
    dur = end - start
    if dur < args.min_clip:
        end = start + args.min_clip
        dur = args.min_clip
    out = Path(args.root_out) / f"{src.stem}_track{e['tid']}_s{start:.2f}_e{end:.2f}_p{e['max_p']:.3f}.mp4"
    cmd = ["ffmpeg","-y","-ss",str(start),"-i",str(src),"-t",str(dur),"-vf","fps=10,scale=640:-2","-an",str(out)]
    print("->", out, "dur", dur)
    subprocess.run(cmd, check=False)
print("done")
