#!/usr/bin/env python3
import os, subprocess

BAD_LIST = "output/bad_demos/list.txt"
OUT_DIR = "output/bad_demos/fixed"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(BAD_LIST):
    print("No bad list found. Run tools/find_bad_demos.py first.")
    raise SystemExit

fixed = 0
with open(BAD_LIST, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        reason, src = parts
        if not os.path.exists(src):
            continue
        name = os.path.basename(src)
        dst = os.path.join(OUT_DIR, name)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", src,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            dst,
        ]
        print("Fixing:", src)
        try:
            subprocess.run(cmd, check=True)
            fixed += 1
        except Exception as e:
            print("Failed:", src, e)

print("Fixed", fixed, "clips into", OUT_DIR)
