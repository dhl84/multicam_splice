"""FCPXML emitter for a multi-angle splice.

A `Sequence` is a flat, frame-accurate plan: an ordered spine of `VideoClip` and
`Dissolve` items, each `VideoClip` optionally hosting connected `AudioClip`s
(e.g. a continuous audio bed on a lower lane). This module turns that plan into a
DTD-valid FCPXML and can validate it against Apple's bundled DTD.

Everything is in integer frames at the sequence rate, so segments tile with no
gaps or overlaps. Self-contained: no third-party imports.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote
from urllib.request import pathname2url

FCPXML_VERSION = "1.9"
# Cross Dissolve / Audio Crossfade ids taken from a real Final Cut Pro export.
_CROSS_DISSOLVE_UID = "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"


# --- time helpers ---------------------------------------------------------

class T:
    def __init__(self, fps_num: int, fps_den: int):
        self.n, self.d = fps_num, fps_den

    def frame_dur(self) -> str:
        return f"{self.d}/{self.n}s"

    def f(self, count: int) -> str:
        v = count * self.d
        return f"{v}/{self.n}s" if v else "0s"

    def to_frames(self, sec: float) -> int:
        return round(sec * self.n / self.d)


def file_url(p: Path) -> str:
    url = "file://" + pathname2url(str(p.resolve()))
    if Path(unquote(url[len("file://"):])) != p.resolve():
        raise ValueError(f"file URL does not round-trip for {p}: {url}")
    return url


def parse_timecode(tc: str, fps_num: int, fps_den: int) -> tuple[int, bool]:
    """Embedded timecode 'HH:MM:SS:FF' (';' before FF = drop-frame) -> (frame, df)."""
    if not tc:
        return 0, False
    drop = ";" in tc
    h, m, s, f = (int(x) for x in re.split("[:;]", tc.strip()))
    fps_round = round(fps_num / fps_den)
    if drop:
        dropn = fps_round // 15
        total_min = 60 * h + m
        fn = ((h * 3600 + m * 60 + s) * fps_round + f
              - dropn * (total_min - total_min // 10))
    else:
        fn = (h * 3600 + m * 60 + s) * fps_round + f
    return fn, drop


# --- plan model -----------------------------------------------------------

@dataclass
class Asset:
    id: str
    name: str
    path: Path
    start_f: int                 # media start (timecode) in frames
    dur_f: int
    tcfmt: str = "NDF"
    has_audio: bool = True       # emit audio metadata only when the source has it
    audio_channels: int = 0      # actual channel count from the probe (0 = unknown)
    audio_rate: int = 0          # actual sample rate (Hz) from the probe (0 = unknown)


@dataclass
class AudioClip:
    """A connected clip (audio only by default), e.g. the continuous bed."""
    ref: str
    name: str
    tl_off_f: int                # absolute timeline offset (frames)
    dur_f: int
    media_in_f: int
    lane: int = -1
    fade_in_f: int = 0
    fade_out_f: int = 0
    src: str = "audio"
    role: str = "dialogue"


@dataclass
class VideoClip:
    ref: str
    name: str
    tl_off_f: int                # absolute timeline offset (frames)
    dur_f: int
    media_in_f: int
    tcfmt: str = "NDF"
    src: str = "all"             # 'all' keeps own audio; 'video' = picture only
    mute: bool = False           # force own audio to -96dB (reliable across FCP)
    opacity_fade_out_f: int = 0  # fade picture to black over the last N frames
    punch_scale: float = 0.0     # >1: slow push-in from 1.0 to this over the clip
    anchors: list[AudioClip] = field(default_factory=list)


@dataclass
class Dissolve:
    tl_off_f: int                # transition centre-left edge (absolute frames)
    dur_f: int


@dataclass
class Sequence:
    fps_num: int
    fps_den: int
    width: int
    height: int
    items: list                  # ordered VideoClip | Dissolve
    assets: list[Asset]
    project_name: str = "Multicam splice"
    event_name: str = "Multicam"
    color_space: str = "1-1-1 (Rec. 709)"


# --- emitter --------------------------------------------------------------

def _fade(parent: ET.Element, kind: str, frames: int, t: T, ftype: str):
    """Append a fadeIn/fadeOut animating `parent`'s 'amount' param."""
    p = parent.find("param[@name='amount']")
    if p is None:
        p = ET.SubElement(parent, "param", name="amount")
    ET.SubElement(p, kind, type=ftype, duration=t.f(frames))


def build(seq: Sequence, out_path: Path) -> Path:
    t = T(seq.fps_num, seq.fps_den)
    total_f = max((it.tl_off_f + it.dur_f) for it in seq.items
                  if isinstance(it, VideoClip))

    root = ET.Element("fcpxml", version=FCPXML_VERSION)
    resources = ET.SubElement(root, "resources")
    ET.SubElement(resources, "format", id="r1",
                  name=f"FFVideoFormat{seq.height}p{seq.fps_num // 1000}",
                  frameDuration=t.frame_dur(), width=str(seq.width),
                  height=str(seq.height), colorSpace=seq.color_space)
    if any(isinstance(it, Dissolve) for it in seq.items):
        ET.SubElement(resources, "effect", id="rDis", name="Cross Dissolve",
                      uid=_CROSS_DISSOLVE_UID)
    for a in seq.assets:
        attrs = {"id": a.id, "name": a.name, "start": t.f(a.start_f),
                 "duration": t.f(a.dur_f), "hasVideo": "1", "format": "r1",
                 "videoSources": "1"}
        # Only claim audio when the source actually has it, and report its real
        # channel count / sample rate — never a fabricated stereo/48 kHz. When a
        # field is unknown (0) it is omitted rather than guessed.
        if a.has_audio:
            attrs.update(hasAudio="1", audioSources="1")
            if a.audio_channels > 0:
                attrs["audioChannels"] = str(a.audio_channels)
            if a.audio_rate > 0:
                attrs["audioRate"] = str(a.audio_rate)
        else:
            attrs["hasAudio"] = "0"
        asset = ET.SubElement(resources, "asset", **attrs)
        ET.SubElement(asset, "media-rep", kind="original-media",
                      src=file_url(a.path))

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=seq.event_name)
    project = ET.SubElement(event, "project", name=seq.project_name)
    sequence = ET.SubElement(project, "sequence", format="r1",
                             duration=t.f(total_f), tcStart="0s",
                             tcFormat="NDF", audioLayout="stereo", audioRate="48k")
    spine = ET.SubElement(sequence, "spine")

    for it in seq.items:
        if isinstance(it, Dissolve):
            tr = ET.SubElement(spine, "transition", name="Cross Dissolve",
                               offset=t.f(it.tl_off_f), duration=t.f(it.dur_f))
            fv = ET.SubElement(tr, "filter-video", ref="rDis", name="Cross Dissolve")
            ET.SubElement(fv, "param", name="Look", key="1", value="11 (Video)")
            ET.SubElement(fv, "param", name="Amount", key="2", value="100")
            ET.SubElement(fv, "param", name="Ease", key="50", value="2 (In & Out)")
            continue

        clip = ET.SubElement(spine, "asset-clip", ref=it.ref,
                             offset=t.f(it.tl_off_f), name=it.name,
                             duration=t.f(it.dur_f), start=t.f(it.media_in_f),
                             tcFormat=it.tcfmt, srcEnable=it.src)
        # intrinsic-params: video (transform, then opacity fade) THEN audio (mute)
        # per the DTD. Keyframe times are in the clip's source timebase, the same
        # coordinate as its `start`.
        if it.punch_scale > 1.0:
            at = ET.SubElement(clip, "adjust-transform")
            ps = ET.SubElement(at, "param", name="scale")
            ka = ET.SubElement(ps, "keyframeAnimation")
            ET.SubElement(ka, "keyframe", time=t.f(it.media_in_f), value="1 1")
            ET.SubElement(ka, "keyframe", time=t.f(it.media_in_f + it.dur_f),
                          value=f"{it.punch_scale:g} {it.punch_scale:g}")
        if it.opacity_fade_out_f:
            ab = ET.SubElement(clip, "adjust-blend")
            _fade(ab, "fadeOut", it.opacity_fade_out_f, t, "easeInOut")
        if it.mute:
            ET.SubElement(clip, "adjust-volume", amount="-96dB")
        # connected clips (audio bed): offset is in the host's local time scale
        for ac in it.anchors:
            local_off = it.media_in_f + (ac.tl_off_f - it.tl_off_f)
            sub = ET.SubElement(clip, "asset-clip", ref=ac.ref, lane=str(ac.lane),
                                offset=t.f(local_off), name=ac.name,
                                duration=t.f(ac.dur_f), start=t.f(ac.media_in_f),
                                tcFormat="NDF", srcEnable=ac.src,
                                audioRole=ac.role)
            if ac.fade_in_f or ac.fade_out_f:
                av = ET.SubElement(sub, "adjust-volume")
                if ac.fade_in_f:
                    _fade(av, "fadeIn", ac.fade_in_f, t, "easeIn")
                if ac.fade_out_f:
                    _fade(av, "fadeOut", ac.fade_out_f, t, "easeOut")

    _indent(root)
    out_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
                        + ET.tostring(root, encoding="unicode") + "\n",
                        encoding="utf-8")
    return out_path


def _indent(elem, level=0):
    pad = "\n" + "    " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "    "
        for child in elem:
            _indent(child, level + 1)
            if not (child.tail or "").strip():
                child.tail = pad + "    "
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad


def check_media(path: Path) -> list[str]:
    missing = []
    for mr in ET.parse(path).getroot().iter("media-rep"):
        src = mr.get("src", "")
        if src.startswith("file://"):
            fp = Path(unquote(src[len("file://"):]))
            if not fp.exists():
                missing.append(fp.name)
    return missing


def validate(path: Path, version: str = FCPXML_VERSION) -> tuple[bool, str]:
    missing = check_media(path)
    if missing:
        return False, "missing media: " + ", ".join(missing)
    dtd = Path("/Applications/Final Cut Pro.app/Contents/Frameworks/"
               "Interchange.framework/Versions/A/Resources/"
               f"FCPXMLv{version.replace('.', '_')}.dtd")
    if not dtd.exists():
        return True, f"(DTD not found; media OK; skipped DTD check)"
    with tempfile.NamedTemporaryFile(suffix=".dtd", delete=False) as tmp:
        shutil.copyfile(dtd, tmp.name)
        local = tmp.name
    try:
        r = subprocess.run(["xmllint", "--noout", "--dtdvalid", local, str(path)],
                           capture_output=True, text=True)
    finally:
        Path(local).unlink(missing_ok=True)
    return r.returncode == 0, (r.stderr.strip() or "valid (DTD + media)")
