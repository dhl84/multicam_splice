"""Project configuration (JSON-backed dataclasses)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FormatSpec:
    width: int = 3840
    height: int = 2160
    fps_num: int = 60000
    fps_den: int = 1001


@dataclass
class AngleSpec:
    name: str
    files: list[str]
    timecode: str = ""
    offset: float | None = None      # manual reference-time offset; None = auto-sync


@dataclass
class Segment:
    type: str                         # "intro" | "remix"
    # intro:
    angle: str = ""
    audio: str = "own"                # intro audio source ("own")
    # remix:
    window: list | None = None        # [start_s, end_s] in reference time; None = full
    angles: list[str] = field(default_factory=list)
    audio_bed: str = "auto"           # "auto" or an angle name
    pattern: list[float] = field(default_factory=lambda: [3.5, 5, 2.5, 4, 6, 3, 4.5, 2.5, 5.5, 3.5])
    tail_single_angle: bool = True    # calm single-angle shot where only one covers the tail


@dataclass
class Project:
    media_dir: Path
    output: Path
    angles: dict[str, AngleSpec]
    segments: list[Segment]
    reference: str
    format: FormatSpec = field(default_factory=FormatSpec)
    project_name: str = "Multicam splice"
    event_name: str = "Multicam"
    transition_seconds: float = 1.0   # cross dissolve between segments (0 = hard cut)
    end_fade_seconds: float = 2.5     # fade to black + audio at the very end
    bed_fade_in_seconds: float = 1.0  # bed fade-in at a segment boundary


def load(path: str | Path) -> Project:
    d = json.loads(Path(path).read_text())
    base = Path(d["media_dir"]).expanduser()
    angles = {n: AngleSpec(name=n, files=a["files"], timecode=a.get("timecode", ""),
                           offset=a.get("offset"))
              for n, a in d["angles"].items()}
    segs = [Segment(**s) for s in d["segments"]]
    out = Path(d["output"]).expanduser()
    if not out.is_absolute():
        out = base / out
    return Project(
        media_dir=base, output=out, angles=angles, segments=segs,
        reference=d["reference"], format=FormatSpec(**d.get("format", {})),
        project_name=d.get("project_name", "Multicam splice"),
        event_name=d.get("event_name", "Multicam"),
        transition_seconds=d.get("transition_seconds", 1.0),
        end_fade_seconds=d.get("end_fade_seconds", 2.5),
        bed_fade_in_seconds=d.get("bed_fade_in_seconds", 1.0),
    )
