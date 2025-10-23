import argparse, os
from pathlib import Path
import cv2
import glob

def write_video(frames, out_path, fps=10):
    if not frames:
        return False
    img0 = cv2.imread(frames[0])
    if img0 is None:
        return False
    h, w = img0.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    ok = True
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            ok = False
            break
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    dirs = [p for p in root.rglob("*") if p.is_dir() and "Test" in p.parts]
    dirs = sorted(dirs)[:args.limit]
    converted = 0
    for d in dirs:
        frames = sorted(glob.glob(str(d / "*.tif")))
        if not frames:
            continue
        rel = d.relative_to(root)
        out_dir = out_root / rel
        out_mp4 = out_dir / "sequence.mp4"
        if write_video(frames, out_mp4, fps=args.fps):
            converted += 1
            print(f"[OK] {rel} -> {out_mp4}")
        else:
            print(f"[SKIP] {rel} (no frames or read error)")
    print(f"Done. Converted {converted} folders.")

if __name__ == "__main__":
    main()
