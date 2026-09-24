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
    channels: int = 0           # audio channel count (0 = no audio)
    avg_fps_num: int = 0        # avg_frame_rate (total frames / duration)
    avg_fps_den: int = 1        # — differs from r_frame_rate on VFR media

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def avg_fps(self) -> float:
        return self.avg_fps_num / self.avg_fps_den if self.avg_fps_den else 0.0

    @property
    def is_vfr(self) -> bool:
        """True when the average frame rate diverges from the nominal
        (r_frame_rate) by more than 1% — the standard ffprobe signal for
        variable-frame-rate media (common on iPhone capture). Such media cannot
        be timed by frame count against a fixed project rate without drift."""
        if not self.avg_fps or not self.fps:
            return False
        return abs(self.avg_fps / self.fps - 1.0) > 0.01


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
    anum, aden = (int(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
    if aden == 0:
        anum, aden = num, den
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
        channels=int(a.get("channels", 0)) if a else 0,
        avg_fps_num=anum, avg_fps_den=aden,
    )


def validate_format(infos: list[MediaInfo], fps_num: int, fps_den: int,
                    width: int, height: int) -> None:
    """Reject media that cannot share the project's single sequence format.

    The emitter declares one `format` and times every clip by frame count at
    the project rate, so each input must be CFR at the project frame rate and
    the project raster. Mismatches here would silently drift or distort, so we
    fail early with an actionable message naming the file and the conflict.
    """
    proj_fps = fps_num / fps_den
    problems: list[str] = []
    for inf in infos:
        nm = inf.path.name
        if inf.is_vfr:
            problems.append(
                f"{nm}: variable frame rate (nominal {inf.fps:.3f}, "
                f"average {inf.avg_fps:.3f} fps). Transcode to constant frame "
                f"rate first, e.g. ffmpeg -i {nm} -vsync cfr -r "
                f"{proj_fps:.3f} out.mp4")
        elif abs(inf.fps / proj_fps - 1.0) > 1e-4:
            problems.append(
                f"{nm}: frame rate {inf.fps:.3f} fps "
                f"({inf.fps_num}/{inf.fps_den}) does not match the project "
                f"{proj_fps:.3f} fps ({fps_num}/{fps_den}). Conform it first "
                f"or set the project format to match.")
        if (inf.width, inf.height) != (width, height):
            problems.append(
                f"{nm}: raster {inf.width}x{inf.height} does not match the "
                f"project {width}x{height}. Mixed frame sizes are not "
                f"supported; scale/crop it first or change the project format.")
    if problems:
        raise ValueError(
            "incompatible source media for this project format:\n  - "
            + "\n  - ".join(problems))
