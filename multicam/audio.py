"""Pick the cleanest audio source for an audio bed.

'Clean' here = loud enough but not clipping. We score each candidate from
ffmpeg's volumedetect (mean level, and how many samples are pinned at 0 dBFS =
clipping) and prefer the source with the least clipping at a healthy level.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioStats:
    name: str
    mean_db: float
    max_db: float
    clip_samples: int      # samples at 0 dBFS (histogram_0db)

    @property
    def score(self) -> float:
        # lower is better: clipping dominates, then distance from a -18 dB target
        return self.clip_samples * 1.0 + abs(self.mean_db + 18.0) * 0.5


def measure(path: Path, name: str = "") -> AudioStats:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-map", "a:0",
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    mean = mx = 0.0
    clip = 0
    for line in out.splitlines():
        if "mean_volume:" in line:
            mean = float(line.split("mean_volume:")[1].split("dB")[0])
        elif "max_volume:" in line:
            mx = float(line.split("max_volume:")[1].split("dB")[0])
        elif "histogram_0db:" in line:
            clip = int(line.split("histogram_0db:")[1])
    return AudioStats(name or Path(path).name, mean, mx, clip)


def pick_cleanest(paths: dict[str, Path]) -> tuple[str, list[AudioStats]]:
    """Return (best_name, all_stats). `paths` maps source name -> a file."""
    stats = [measure(p, n) for n, p in paths.items()]
    best = min(stats, key=lambda s: s.score)
    return best.name, stats
