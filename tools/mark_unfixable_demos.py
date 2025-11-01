#!/usr/bin/env python3
import os, shutil

BAD_LIST = "output/bad_demos/list.txt"
FIXED_DIR = "output/bad_demos/fixed"
UNFIXABLE_DIR = "output/bad_demos/unfixable"

os.makedirs(UNFIXABLE_DIR, exist_ok=True)

if not os.path.exists(BAD_LIST):
    print("[ERR] output/bad_demos/list.txt not found — run tools/find_bad_demos.py first.")
    raise SystemExit

moved = 0
with open(BAD_LIST, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # lines look like:
        # BAD: not opened -> output/demos/....
        if "->" not in line:
            continue
        _, path = line.split("->", 1)
        path = path.strip()
        name = os.path.basename(path)
        fixed_path = os.path.join(FIXED_DIR, name)
        if not os.path.exists(fixed_path):
            # ffmpeg failed or we never fixed it → move original to unfixable
            if os.path.exists(path):
                dst = os.path.join(UNFIXABLE_DIR, name)
                shutil.move(path, dst)
                print("[UNFIXABLE] moved", path, "→", dst)
                moved += 1

print(f"[DONE] moved {moved} unfixable clips to {UNFIXABLE_DIR}")
