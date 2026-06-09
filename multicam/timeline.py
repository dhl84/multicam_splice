"""Turn a Project config into a frame-accurate FCPXML `Sequence` plan.

Pipeline: probe every file -> sync the remix angles onto the reference timeline
(auto unless a manual offset is given) -> lay out each segment on a single spine
(an `intro` plays one angle straight with its own audio; a `remix` flicks between
angles over a continuous, single-source audio bed) -> insert cross dissolves
between segments and a fade-to-black at the end.

The key audio guarantee: every remix clip's own camera audio is force-muted
(srcEnable=video AND adjust-volume -96 dB), so only the chosen bed is heard --
one clean source, never a jumble of per-shot audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import audio, fcpxml
from .config import Project, Segment
from .probe import MediaInfo, probe
from .sync import spectral_flux, sync_to_reference, SyncResult


@dataclass
class _Angle:
    name: str
    paths: list[Path]
    infos: list[MediaInfo]
    asset_ids: list[str]
    start_fs: list[int]          # per-file media start (timecode) frames
    cum_frames: list[int]        # cumulative frame offset of each file
    offset: float                # reference-time of this angle's t=0
    total_frames: int

    def covers(self, t0: float, t1: float, fps: float, margin: float = 0.4) -> bool:
        lo = self.offset + margin
        hi = self.offset + self.total_frames / fps - margin
        return lo <= t0 and t1 <= hi

    def media_at(self, ref_t: float, fps: float) -> tuple[str, int]:
        """(asset_id, media_in_frame) for showing reference-time ref_t."""
        local_f = round((ref_t - self.offset) * fps)
        local_f = max(0, min(local_f, self.total_frames - 1))
        i = max(j for j in range(len(self.cum_frames)) if self.cum_frames[j] <= local_f)
        return self.asset_ids[i], self.start_fs[i] + (local_f - self.cum_frames[i])


class Builder:
    def __init__(self, proj: Project, log=print):
        self.p = proj
        self.log = log
        self.fps = proj.format.fps_num / proj.format.fps_den
        self.fps_n, self.fps_d = proj.format.fps_num, proj.format.fps_den
        self._aid = 0
        self.angles: dict[str, _Angle] = {}
        self.assets: list[fcpxml.Asset] = []

    def f(self, sec: float) -> int:
        return round(sec * self.fps_n / self.fps_d)

    # --- setup: probe + register assets + sync -----------------------------

    def _register(self, name: str) -> _Angle:
        if name in self.angles:
            return self.angles[name]
        spec = self.p.angles[name]
        paths = [self.p.media_dir / fn for fn in spec.files]
        infos = [probe(pp) for pp in paths]
        ids, starts, cum = [], [], []
        running = 0
        for inf in infos:
            self._aid += 1
            aid = f"v{self._aid}"
            ids.append(aid)
            sf, drop = fcpxml.parse_timecode(spec.timecode if inf is infos[0] else "",
                                             self.fps_n, self.fps_d)
            starts.append(sf)
            cum.append(running)
            running += inf.n_frames
            self.assets.append(fcpxml.Asset(
                id=aid, name=name, path=inf.path, start_f=sf, dur_f=inf.n_frames,
                tcfmt="DF" if drop else "NDF"))
        ang = _Angle(name=name, paths=paths, infos=infos, asset_ids=ids,
                     start_fs=starts, cum_frames=cum, offset=0.0,
                     total_frames=running)
        self.angles[name] = ang
        return ang

    def setup(self):
        ref = self._register(self.p.reference)
        ref.offset = 0.0
        # which angles are used in remix segments (need syncing)?
        remix_angles = {a for s in self.p.segments if s.type == "remix" for a in s.angles}
        ref_flux = None
        for name in remix_angles:
            if name == self.p.reference:
                continue
            ang = self._register(name)
            spec = self.p.angles[name]
            if spec.offset is not None:
                ang.offset = spec.offset
                self.log(f"sync {name}: offset {ang.offset:+.3f}s (manual)")
                continue
            if ref_flux is None:
                self.log("computing reference spectral flux...")
                ref_flux = spectral_flux(ref.paths)
            r: SyncResult = sync_to_reference(ref.paths, ang.paths, ref_flux=ref_flux)
            ang.offset = r.offset
            tag = "CONFIDENT" if r.confident else "LOW CONFIDENCE — verify/override"
            self.log(f"sync {name}: offset {r.offset:+.3f}s  PSR {r.psr:.1f}  [{tag}]")
        # intro angles just need registering
        for s in self.p.segments:
            if s.type == "intro":
                self._register(s.angle)

    # --- audio bed ---------------------------------------------------------

    def _bed_spans(self, seg: Segment, ws: float, we: float):
        """Spans [(angle, a, b)] tiling [ws,we], each from the cleanest covering
        source (priority: requested bed, else by measured cleanliness)."""
        cands = list(seg.angles)
        if seg.audio_bed != "auto" and seg.audio_bed in cands:
            cands.remove(seg.audio_bed)
            cands.insert(0, seg.audio_bed)
        else:
            stats = {n: audio.measure(self.angles[n].paths[0], n) for n in cands}
            cands.sort(key=lambda n: stats[n].score)
            self.log("audio bed cleanliness (lower=better): "
                     + ", ".join(f"{n} {stats[n].score:.0f}"
                                 f"(clip {stats[n].clip_samples})" for n in cands))
        self.log(f"audio bed priority: {cands}")

        spans, t, step = [], ws, 0.2
        while t < we - 1e-6:
            pick = next((n for n in cands
                         if self.angles[n].covers(t, min(t + step, we), self.fps)), None)
            if pick is None:               # nobody covers: keep nearest cleanest
                pick = cands[0]
            b = t
            while b < we - 1e-6 and (
                    next((n for n in cands
                          if self.angles[n].covers(b, min(b + step, we), self.fps)),
                         cands[0]) == pick):
                b += step
            spans.append((pick, t, min(b, we)))
            t = min(b, we)
        # merge adjacent same-source spans
        merged = [spans[0]]
        for s in spans[1:]:
            if s[0] == merged[-1][0]:
                merged[-1] = (merged[-1][0], merged[-1][1], s[2])
            else:
                merged.append(s)
        return merged

    # --- build -------------------------------------------------------------

    def build(self) -> fcpxml.Sequence:
        self.setup()
        items: list = []
        cursor = 0
        D = self.f(self.p.transition_seconds) if self.p.transition_seconds > 0 else 0
        n_seg = len(self.p.segments)

        for si, seg in enumerate(self.p.segments):
            seg_start = cursor
            seg_clips: list[fcpxml.VideoClip] = []
            is_last = si == n_seg - 1

            if seg.type == "intro":
                ang = self.angles[seg.angle]
                for k, (aid, inf, sf) in enumerate(
                        zip(ang.asset_ids, ang.infos, ang.start_fs)):
                    c = fcpxml.VideoClip(
                        ref=aid, name=f"{seg.angle} (intro)", tl_off_f=cursor,
                        dur_f=inf.n_frames, media_in_f=sf,
                        tcfmt=self.assets_by_id(aid).tcfmt, src="all")
                    items.append(c); seg_clips.append(c)
                    cursor += inf.n_frames

            elif seg.type == "remix":
                ws, we = (seg.window if seg.window
                          else [self.angles[seg.angle or seg.angles[0]].offset,
                                self._full_end(seg)])
                shots = self._shots(seg, ws, we)
                first = None
                for (aname, t0, t1) in shots:
                    ang = self.angles[aname]
                    aid, mi = ang.media_at(t0, self.fps)
                    off = seg_start + (self.f(t0) - self.f(ws))
                    dur = self.f(t1) - self.f(t0)
                    if dur <= 0:
                        continue
                    c = fcpxml.VideoClip(
                        ref=aid, name=aname, tl_off_f=off, dur_f=dur, media_in_f=mi,
                        src="video", mute=True)
                    items.append(c); seg_clips.append(c)
                    first = first or c
                    cursor = off + dur
                # single-source audio bed across [ws, we], anchored to first clip
                bed_in = self.f(self.p.bed_fade_in_seconds) if si > 0 else 0
                bed_out = self.f(self.p.end_fade_seconds) if is_last else 0
                spans = self._bed_spans(seg, ws, we)
                for j, (bname, a, b) in enumerate(spans):
                    bang = self.angles[bname]
                    aid, mi = bang.media_at(a, self.fps)
                    off = seg_start + (self.f(a) - self.f(ws))
                    first.anchors.append(fcpxml.AudioClip(
                        ref=aid, name=f"{bname} audio (bed)", tl_off_f=off,
                        dur_f=self.f(b) - self.f(a), media_in_f=mi,
                        fade_in_f=bed_in if j == 0 else 0,
                        fade_out_f=bed_out if j == len(spans) - 1 else 0))

            # transition into the NEXT segment: trim this segment's tail for a handle
            if D and not is_last:
                last = seg_clips[-1]
                if last.dur_f > D + 2:
                    last.dur_f -= D
                    cursor -= D
                    items.append(fcpxml.Dissolve(tl_off_f=cursor - D // 2, dur_f=D))

        # fade the very last picture to black
        last_video = next(it for it in reversed(items)
                          if isinstance(it, fcpxml.VideoClip))
        last_video.opacity_fade_out_f = self.f(self.p.end_fade_seconds)

        return fcpxml.Sequence(
            fps_num=self.fps_n, fps_den=self.fps_d,
            width=self.p.format.width, height=self.p.format.height,
            items=items, assets=self.assets,
            project_name=self.p.project_name, event_name=self.p.event_name)

    # --- helpers -----------------------------------------------------------

    def assets_by_id(self, aid: str) -> fcpxml.Asset:
        return next(a for a in self.assets if a.id == aid)

    def _full_end(self, seg: Segment) -> float:
        return max(self.angles[a].offset + self.angles[a].total_frames / self.fps
                   for a in seg.angles)

    def _shots(self, seg: Segment, ws: float, we: float):
        """Alternate angles with varied lengths; calm single-angle tail."""
        pat = seg.pattern
        # last reference-time at which more than one angle still covers
        multi_end = we
        if seg.tail_single_angle and len(seg.angles) > 1:
            ends = sorted(self.angles[a].offset + self.angles[a].total_frames / self.fps
                          for a in seg.angles)
            multi_end = min(we, ends[-2] - 0.4)        # 2nd-latest angle runs out
        shots, t, i, cur = [], ws, 0, 0
        order_base = list(seg.angles)
        alt_end = min(we, multi_end)
        while t < alt_end - 1e-6:
            t1 = min(t + pat[i % len(pat)], alt_end)
            order = [order_base[(cur + k) % len(order_base)] for k in range(len(order_base))]
            pick = next((a for a in order if self.angles[a].covers(t, t1, self.fps)),
                        order_base[0])
            shots.append((pick, t, t1))
            cur = (order_base.index(pick) + 1) % len(order_base)
            t = t1; i += 1
        if we - alt_end > 0.1:
            # one calm shot from whichever angle covers the tail (prefer reference)
            tail = next((a for a in [self.p.reference] + order_base
                         if a in self.angles
                         and self.angles[a].covers(alt_end, we, self.fps)), order_base[0])
            shots.append((tail, alt_end, we))
        return shots


def render(proj: Project, log=print) -> tuple[Path, bool, str]:
    seq = Builder(proj, log).build()
    fcpxml.build(seq, proj.output)
    ok, msg = fcpxml.validate(proj.output)
    return proj.output, ok, msg
