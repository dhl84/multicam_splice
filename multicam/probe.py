"""ffprobe wrappers: read duration, frame rate, geometry, timecode, audio."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MediaInfo:
    path: Path
    duration: float
    fps_num: int
    fps_den: int
    width: int
    height: int
    n_frames: int
    timecode: str          # embedded start timecode, "" if none
    sample_rate: int
    has_audio: bool

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def probe(path: Path) -> MediaInfo:
    path = Path(path)
    info = _ffprobe(path)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if v is None:
        raise ValueError(f"{path.name}: no video stream")

    num, den = (int(x) for x in v.get("r_frame_rate", "0/1").split("/"))
    duration = float(info["format"]["duration"])

    # embedded timecode: try video/data streams, then format tag
    tc = ""
    for s in info["streams"]:
        tc = (s.get("tags", {}) or {}).get("timecode", "")
        if tc:
            break
    if not tc:
        tc = (info["format"].get("tags", {}) or {}).get("timecode", "")

    n_frames = int(v.get("nb_frames") or round(duration * num / den))
    return MediaInfo(
        path=path, duration=duration, fps_num=num, fps_den=den,
        width=int(v["width"]), height=int(v["height"]), n_frames=n_frames,
        timecode=tc, has_audio=a is not None,
        sample_rate=int(a["sample_rate"]) if a else 0,
    )
