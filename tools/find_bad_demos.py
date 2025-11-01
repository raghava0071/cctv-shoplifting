#!/usr/bin/env python3
import os, cv2, glob

DEMOS = "output/demos"
bad = []
good = 0

for path in sorted(glob.glob(os.path.join(DEMOS, "*.mp4"))):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        bad.append((path, "not opened"))
        continue
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 1:
        bad.append((path, f"{n} frames"))
    else:
        good += 1

print("Good videos:", good)
print("Bad videos:", len(bad))
for p, reason in bad[:50]:
    print("BAD:", reason, "->", p)

if bad:
    os.makedirs("output/bad_demos", exist_ok=True)
    with open("output/bad_demos/list.txt", "w") as f:
        for p, reason in bad:
            f.write(f"{reason}\t{p}\n")
    print("Wrote bad list to output/bad_demos/list.txt")
