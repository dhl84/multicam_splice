"""Pick the cleanest audio source for an audio bed.

'Clean' here = a healthy, broadcast-sane level with headroom and stable
dynamics. We measure each candidate with ffmpeg's `loudnorm` in EBU R128
analysis mode — integrated loudness (LUFS), true-peak (dBTP), and loudness
range (LRA) — and prefer the source closest to a comfortable level that isn't
near clipping and isn't wildly dynamic. (This replaced a cruder mean-volume +
0-dBFS-sample-count heuristic: true-peak catches inter-sample clipping that a
raw sample histogram misses, and LUFS tracks perceived level, not raw RMS.)
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

# A comfortable bed loudness to aim for; sources near this score best.
_TARGET_LUFS = -18.0


@dataclass
class AudioStats:
    name: str
    lufs: float            # integrated loudness, EBU R128 (negative)
    true_peak: float       # dBTP — how close to clipping (0 = at the ceiling)
    lra: float             # loudness range, LU (higher = more dynamic/unstable)

    @property
    def score(self) -> float:
        # lower is cleaner. Clipping risk dominates (true-peak above -1 dBTP),
        # then distance from a healthy level, then excessive dynamic range.
        clip_pen = max(0.0, self.true_peak + 1.0) * 4.0
        level_pen = abs(self.lufs - _TARGET_LUFS) * 0.5
        range_pen = max(0.0, self.lra - 12.0) * 0.2
        return clip_pen + level_pen + range_pen


def measure(path: Path, name: str = "") -> AudioStats:
    """EBU R128 stats for a source's first audio track via ffmpeg loudnorm.
    A track with no/again-unreadable audio scores as effectively silent."""
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "a:0",
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    lufs, tp, lra = -70.0, -70.0, 0.0     # silence/unreadable -> never the bed
    a, b = err.rfind("{"), err.rfind("}")
    if a != -1 and b > a:
        try:
            d = json.loads(err[a:b + 1])
            lufs, tp, lra = (float(d["input_i"]), float(d["input_tp"]),
                             float(d["input_lra"]))
        except (ValueError, KeyError):
            pass
    return AudioStats(name or Path(path).name, lufs, tp, lra)


def pick_cleanest(paths: dict[str, Path]) -> tuple[str, list[AudioStats]]:
    """Return (best_name, all_stats). `paths` maps source name -> a file."""
    stats = [measure(p, n) for n, p in paths.items()]
    best = min(stats, key=lambda s: s.score)
    return best.name, stats
