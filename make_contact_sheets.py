"""Extract frames from a video and write batched contact-sheet JPEGs.

Usage:
  python make_contact_sheets.py <video_file_or_files> <output_dir> [--fps 0.5] [--max-seconds 500]

Each sheet contains up to 20 frames (4 cols × 5 rows) with the local timestamp
printed on each thumbnail.  Sheets are named sheet_0000.jpg, sheet_0001.jpg, …
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 4, 5
PER_SHEET = COLS * ROWS
THUMB_W, THUMB_H = 320, 180
LABEL_H = 24


def extract_jpegs(files, fps, max_seconds):
    frames = []
    timestamps = []
    remaining = max_seconds
    t_offset = 0.0
    for p in files:
        if remaining is not None and remaining <= 0:
            break
        with tempfile.TemporaryDirectory() as tmp:
            cmd = ["ffmpeg", "-v", "error", "-i", str(p)]
            if remaining is not None:
                cmd += ["-t", f"{remaining:.3f}"]
            cmd += ["-vf", f"scale={THUMB_W}:{THUMB_H},fps={fps}",
                    "-q:v", "5", os.path.join(tmp, "frame%06d.jpg")]
            subprocess.run(cmd, check=True, capture_output=True)
            jpegs = sorted(glob.glob(os.path.join(tmp, "frame*.jpg")))
            for i, j in enumerate(jpegs):
                frames.append(Path(j).read_bytes())
                timestamps.append(t_offset + i / fps)
            count = len(jpegs)
        t_offset += count / fps
        if remaining is not None:
            remaining -= count / fps
    return frames, timestamps


def make_sheets(frames, timestamps, out_dir, prefix="sheet"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sheet_idx = 0
    for start in range(0, len(frames), PER_SHEET):
        batch = frames[start:start + PER_SHEET]
        tss = timestamps[start:start + PER_SHEET]
        n = len(batch)
        rows_used = math.ceil(n / COLS)
        img = Image.new("RGB", (COLS * THUMB_W, rows_used * (THUMB_H + LABEL_H)),
                        color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except Exception:
            font = ImageFont.load_default()
        for idx, (jpeg_bytes, ts) in enumerate(zip(batch, tss)):
            row, col = divmod(idx, COLS)
            x, y = col * THUMB_W, row * (THUMB_H + LABEL_H)
            import io
            thumb = Image.open(io.BytesIO(jpeg_bytes))
            img.paste(thumb, (x, y))
            label = f"#{start + idx + 1}  t={ts:.1f}s"
            draw.rectangle([x, y + THUMB_H, x + THUMB_W, y + THUMB_H + LABEL_H],
                           fill=(0, 0, 0))
            draw.text((x + 4, y + THUMB_H + 4), label, fill=(255, 255, 0), font=font)
        out = Path(out_dir) / f"{prefix}_{sheet_idx:04d}.jpg"
        img.save(out, quality=85)
        print(f"  wrote {out}  (frames {start + 1}–{start + len(batch)})")
        sheet_idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="/tmp/contact_sheets")
    ap.add_argument("--fps", type=float, default=0.5)
    ap.add_argument("--max-seconds", type=float, default=500.0)
    ap.add_argument("--prefix", default="sheet")
    args = ap.parse_args()
    paths = [Path(f) for f in args.files]
    print(f"Extracting from {[p.name for p in paths]} at {args.fps}fps …")
    frames, timestamps = extract_jpegs(paths, args.fps, args.max_seconds)
    print(f"Extracted {len(frames)} frames")
    make_sheets(frames, timestamps, args.out, prefix=args.prefix)
    print("Done.")


if __name__ == "__main__":
    main()
