"""Vision-based frame content classification: athletes vs coach vs empty.

Pre-computes a per-sample label array for each angle using the Claude vision
API at a low frame rate (one sample every 2 s by default). Classification
happens once at setup time and is stored per-angle for use in shot selection.

Labels:
  A — athletes are clearly visible (people competing / doing the workout)
  C — ONLY the coach/trainer is visible; no competing athletes in frame
  E — empty, no people, or too blurry to classify
"""
from __future__ import annotations

import base64
import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CLASSIFY_FPS = 0.5     # one sample every 2 s — sufficient for content decisions
_BATCH = 20            # frames per API call
_MODEL = "claude-haiku-4-5-20251001"


def _extract_jpegs(files: list[Path], fps: float,
                   max_seconds: float | None) -> list[bytes]:
    """Return JPEG bytes for each sampled frame across all files."""
    frames: list[bytes] = []
    remaining = max_seconds
    for p in files:
        if remaining is not None and remaining <= 0:
            break
        with tempfile.TemporaryDirectory() as tmp:
            cmd = ["ffmpeg", "-v", "error", "-i", str(p)]
            if remaining is not None:
                cmd += ["-t", f"{remaining:.3f}"]
            cmd += ["-vf", f"scale=320:180,fps={fps}",
                    "-q:v", "5", os.path.join(tmp, "frame%06d.jpg")]
            subprocess.run(cmd, check=True, capture_output=True)
            file_frames = [Path(j).read_bytes()
                           for j in sorted(glob.glob(os.path.join(tmp, "frame*.jpg")))]
        frames.extend(file_frames)
        if remaining is not None:
            remaining -= len(file_frames) / fps
    return frames


def _cache_path(files: list[Path]) -> Path:
    """Deterministic cache file path derived from the first file's location."""
    return files[0].parent / (files[0].stem + "_content_cache.npy")


def classify_content(
    files: list[Path],
    coach_description: str,
    fps: float = CLASSIFY_FPS,
    max_seconds: float | None = None,
    log=print,
) -> np.ndarray:
    """Return int8 array at ``fps`` samples/sec: 1 = athletes visible, 0 = no athletes.

    Calls the Claude vision API to classify each frame.  Falls back to all-ones
    (assume athletes present) on any API failure so shot-selection degrades
    gracefully to the activity-only filter.
    """
    cache = _cache_path(files)
    if cache.exists():
        data = np.load(cache)
        log(f"  loaded content cache ({len(data)} samples) from {cache.name}")
        return data

    import anthropic

    coach_clause = f"The coach wears {coach_description}. " if coach_description else ""
    frames = _extract_jpegs(files, fps, max_seconds)
    if not frames:
        return np.ones(0, dtype=np.int8)

    client = anthropic.Anthropic()
    labels: list[int] = []
    n_batches = (len(frames) + _BATCH - 1) // _BATCH
    log(f"  classifying {len(frames)} frames in {n_batches} API calls …")

    prompt_suffix = (
        "For each numbered frame above, output exactly one label:\n"
        '  "A" — athletes are clearly visible (people competing / doing the workout)\n'
        '  "C" — ONLY the coach/trainer is visible, no competing athletes in frame\n'
        '  "E" — empty, no people, or too blurry to classify\n\n'
        "Reply with ONLY a JSON array of single-character strings, one per frame, "
        'in order. Example for 3 frames: ["A","C","E"]'
    )

    for bi in range(n_batches):
        batch = frames[bi * _BATCH: (bi + 1) * _BATCH]
        content: list[dict] = []
        for idx, jpeg in enumerate(batch):
            content.append({"type": "text", "text": f"Frame {idx + 1}:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(jpeg).decode(),
                },
            })
        content.append({
            "type": "text",
            "text": (
                f"{coach_clause}These are {len(batch)} sequential frames "
                "from a CrossFit competition video.\n\n" + prompt_suffix
            ),
        })
        try:
            msg = client.messages.create(
                model=_MODEL,
                max_tokens=128,
                messages=[{"role": "user", "content": content}],
            )
            parsed = json.loads(msg.content[0].text.strip())
            labels.extend(1 if str(lbl).upper() == "A" else 0 for lbl in parsed)
        except Exception as exc:
            log(f"  [classify] batch {bi + 1}/{n_batches} error: {exc} — assuming athletes present")
            labels.extend([1] * len(batch))

    result = np.array(labels, dtype=np.int8)
    np.save(cache, result)
    log(f"  saved content cache ({len(result)} samples) to {cache.name}")
    return result
