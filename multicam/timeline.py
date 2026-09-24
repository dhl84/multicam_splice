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

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, fcpxml
from .classify import classify_content, CLASSIFY_FPS
from .config import Project, Segment
from .probe import MediaInfo, probe, validate_format
from .sync import (ACT_FPS, onset_threshold, snap_to_onset, spectral_flux,
                   sync_to_reference, video_activity, SyncResult)


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
    activity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    activity_fps: float = ACT_FPS
    content: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    content_fps: float = CLASSIFY_FPS

    def covers(self, t0: float, t1: float, fps: float, margin: float = 0.4) -> bool:
        lo = self.offset + margin
        hi = self.offset + self.total_frames / fps - margin
        return lo <= t0 and t1 <= hi

    def covers_media(self, t0: float, t1: float, fps: float) -> bool:
        """True when the angle's actual media spans [t0, t1] (no safety margin).
        Used to tile a shot across an angle hand-off so no frame is dropped."""
        return self.offset <= t0 and t1 <= self.offset + self.total_frames / fps

    def media_end(self, fps: float) -> float:
        return self.offset + self.total_frames / fps

    def coverage_margin(self, t0: float, t1: float, fps: float,
                        margin: float = 0.4) -> float:
        """Seconds of media headroom around [t0, t1] (min of the lead-in and
        run-out slack). Negative when the angle does not fully cover the window;
        larger is safer for transitions and push-ins."""
        lo = self.offset + margin
        hi = self.offset + self.total_frames / fps - margin
        return min(t0 - lo, hi - t1)

    def mean_content(self, ref_t0: float, ref_t1: float) -> float:
        """Fraction of content-classified samples in [ref_t0, ref_t1] showing athletes.

        Returns NaN ("unknown") when no content data was computed OR the window
        falls past the analysed range. Callers treat NaN as no-opinion — kept
        through the content filter (never dropped on missing data) and scored
        neutrally in ranking (not as best-case). We do NOT reuse the last sample
        for out-of-range windows: a stale tail value would silently drive shot
        selection late in long edits."""
        if len(self.content) == 0:
            return float("nan")
        local_t0 = ref_t0 - self.offset
        i0 = int(local_t0 * self.content_fps)
        if local_t0 < 0 or i0 >= len(self.content):
            return float("nan")
        local_t1 = ref_t1 - self.offset
        i1 = min(max(i0 + 1, int(local_t1 * self.content_fps) + 1), len(self.content))
        return float(self.content[i0:i1].mean())

    def mean_activity(self, ref_t0: float, ref_t1: float) -> float:
        """Average visual motion energy over the reference-time window [ref_t0, ref_t1].

        Returns NaN when no activity data was computed OR the window falls past
        the analysed range — callers treat NaN as "no opinion" (neither excluded
        by nor preferred in the activity filter) rather than reusing a stale
        last sample."""
        if len(self.activity) == 0:
            return float("nan")
        local_t0 = ref_t0 - self.offset
        i0 = int(local_t0 * self.activity_fps)
        if local_t0 < 0 or i0 >= len(self.activity):
            return float("nan")
        local_t1 = ref_t1 - self.offset
        i1 = min(max(i0 + 1, int(local_t1 * self.activity_fps) + 1), len(self.activity))
        return float(self.activity[i0:i1].mean())

    def media_at(self, ref_t: float, fps: float) -> tuple[str, int]:
        """(asset_id, media_in_frame) for showing reference-time ref_t."""
        local_f = round((ref_t - self.offset) * fps)
        local_f = max(0, min(local_f, self.total_frames - 1))
        i = max(j for j in range(len(self.cum_frames)) if self.cum_frames[j] <= local_f)
        return self.asset_ids[i], self.start_fs[i] + (local_f - self.cum_frames[i])


class Builder:
    # Shot-selection tuning. The activity floor gates out static/empty angles;
    # the weights then rank the survivors (higher score = preferred). The repeat
    # penalty grows with how long the current angle has been held, so variety
    # emerges without ever overriding a clearly better-looking shot.
    _MIN_ACTIVITY = 2.0     # motion-energy floor (sensor noise sits ~0.5–1.5)
    _W_ACTIVITY = 1.0       # normalised motion energy in the shot window
    _W_CONTENT = 1.2        # fraction of frames with athletes visible (vision)
    _W_COVERAGE = 0.5       # media headroom around the shot (transition safety)
    # Saturated repeat penalty is deliberately below the activity weight (1.0):
    # it breaks near-ties for variety but can never flip away from a visibly
    # better angle (e.g. activity 10 vs 3 -> 0.7 gap survives a 0.5 penalty).
    _W_REPEAT = 0.5         # max penalty for staying on the angle already on screen
    _REPEAT_FULL_S = 8.0    # run length at which the repeat penalty saturates

    def __init__(self, proj: Project, log=print):
        self.p = proj
        self.log = log
        self.fps = proj.format.fps_num / proj.format.fps_den
        self.fps_n, self.fps_d = proj.format.fps_num, proj.format.fps_den
        self._aid = 0
        self.angles: dict[str, _Angle] = {}
        self.assets: list[fcpxml.Asset] = []
        self._ref_flux: np.ndarray | None = None   # reference onset function
        self._onset_thr: float = float("inf")      # flux level that counts as an onset
        self._audio_cache: dict[Path, audio.AudioStats] = {}  # per-file measure() memo

    def f(self, sec: float) -> int:
        return round(sec * self.fps_n / self.fps_d)

    # --- setup: probe + register assets + sync -----------------------------

    def _register(self, name: str) -> _Angle:
        if name in self.angles:
            return self.angles[name]
        spec = self.p.angles[name]
        paths = [self.p.media_dir / fn for fn in spec.files]
        infos = [probe(pp) for pp in paths]
        # Enforce timebase/raster correctness before any frame-count maths: the
        # whole timeline times clips by frame count at the project rate, which
        # is only sound for CFR media at the project format.
        validate_format(infos, self.fps_n, self.fps_d,
                        self.p.format.width, self.p.format.height)
        ids, starts, cum = [], [], []
        running = 0
        for inf in infos:
            self._aid += 1
            aid = f"v{self._aid}"
            ids.append(aid)
            tc = (spec.timecode if inf is infos[0] else "") or inf.timecode
            sf, drop = fcpxml.parse_timecode(tc, self.fps_n, self.fps_d)
            starts.append(sf)
            cum.append(running)
            running += inf.n_frames
            self.assets.append(fcpxml.Asset(
                id=aid, name=name, path=inf.path, start_f=sf, dur_f=inf.n_frames,
                tcfmt="DF" if drop else "NDF",
                has_audio=inf.has_audio,
                audio_channels=inf.channels,
                audio_rate=inf.sample_rate))
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
        # cut-on-action reuses the reference onset function the sync computed;
        # when every angle was placed manually, compute it here instead.
        if self.p.cut_on_action and remix_angles:
            if ref_flux is None:
                self.log("computing reference spectral flux (cut-on-action)...")
                ref_flux = spectral_flux(ref.paths)
            self._ref_flux = ref_flux
            self._onset_thr = onset_threshold(ref_flux)
        # How far (local seconds, from each angle's own t=0) does shot selection
        # actually query each remix angle?  Analyse exactly that span — plus a
        # small margin — instead of a fixed first-600 s cap, so late windows in a
        # long edit get real samples rather than a stale cached tail.
        analyse_to = self._analysis_extents(remix_angles)
        # Compute video activity for all remix angles (used in shot selection to
        # avoid cutting to angles with no one in frame).
        for name in remix_angles:
            ang = self.angles[name]
            self.log(f"computing video activity for {name} "
                     f"(through {analyse_to[name]:.0f}s)...")
            ang.activity = video_activity(ang.paths, max_seconds=analyse_to[name])
        # Vision-based content classification (requires coach_description in config).
        # Labels each sample 1=athletes or 0=no-athletes so shot selection can
        # deprioritise coach-only frames even when they pass the activity floor.
        if self.p.coach_description:
            for name in remix_angles:
                ang = self.angles[name]
                self.log(f"classifying content for {name} (vision)...")
                ang.content = classify_content(
                    ang.paths, self.p.coach_description,
                    max_seconds=analyse_to[name], log=self.log)
        # intro angles just need registering
        for s in self.p.segments:
            if s.type == "intro":
                self._register(s.angle)

    # --- audio bed ---------------------------------------------------------

    def _measure(self, path: Path, name: str) -> audio.AudioStats:
        if path not in self._audio_cache:
            self._audio_cache[path] = audio.measure(path, name)
        return self._audio_cache[path]

    def _bed_stats(self, name: str, ws: float, we: float) -> audio.AudioStats:
        """Cleanliness of the material an angle actually contributes to the bed
        across [ws, we]. Split recordings span several files with different
        noise/level/clipping, so we measure every file overlapping the window
        and combine by overlap duration — level and dynamics duration-weighted,
        true-peak taken as the worst case (clipping anywhere is disqualifying).
        Falls back to the first file when the angle never overlaps the window."""
        ang = self.angles[name]
        overlaps: list[tuple[audio.AudioStats, float]] = []
        for i, path in enumerate(ang.paths):
            next_cf = (ang.cum_frames[i + 1] if i + 1 < len(ang.cum_frames)
                       else ang.total_frames)
            lo = ang.offset + ang.cum_frames[i] / self.fps
            hi = ang.offset + next_cf / self.fps
            ov = max(0.0, min(we, hi) - max(ws, lo))
            if ov > 0:
                overlaps.append((self._measure(path, f"{name}[{i}]"), ov))
        if not overlaps:
            return self._measure(ang.paths[0], name)
        total = sum(ov for _, ov in overlaps)
        lufs = sum(s.lufs * ov for s, ov in overlaps) / total
        lra = sum(s.lra * ov for s, ov in overlaps) / total
        true_peak = max(s.true_peak for s, _ in overlaps)
        return audio.AudioStats(name, lufs, true_peak, lra)

    def _bed_spans(self, seg: Segment, ws: float, we: float):
        """Spans [(angle, a, b)] tiling [ws,we], each from the cleanest covering
        source (priority: requested bed, else by measured cleanliness)."""
        cands = list(seg.angles)
        if seg.audio_bed != "auto" and seg.audio_bed in cands:
            cands.remove(seg.audio_bed)
            cands.insert(0, seg.audio_bed)
        else:
            stats = {n: self._bed_stats(n, ws, we) for n in cands}
            cands.sort(key=lambda n: stats[n].score)
            self.log("audio bed cleanliness over window (lower=better): "
                     + ", ".join(f"{n} {stats[n].score:.0f} "
                                 f"({stats[n].lufs:.0f} LUFS, TP {stats[n].true_peak:.1f})"
                                 for n in cands))
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

    def _subspans(self, ang: _Angle, ref_a: float, ref_b: float):
        """Yield (asset_id, media_in_f, dur_f, ref_sub_a, ref_sub_b) split at file
        boundaries so no single clip ever references frames past one file's end."""
        for i in range(len(ang.asset_ids)):
            next_cf = (ang.cum_frames[i + 1] if i + 1 < len(ang.cum_frames)
                       else ang.total_frames)
            t_lo = ang.offset + ang.cum_frames[i] / self.fps
            t_hi = ang.offset + next_cf / self.fps
            sub_a = max(ref_a, t_lo)
            sub_b = min(ref_b, t_hi)
            dur_f = self.f(sub_b) - self.f(sub_a)
            if dur_f <= 0:
                continue
            _, mi = ang.media_at(sub_a, self.fps)
            yield ang.asset_ids[i], mi, dur_f, sub_a, sub_b

    def build(self) -> fcpxml.Sequence:
        self.setup()
        self._validate_coverage()      # reject uncovered remix windows up front
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
                ws, we = self._window(seg)
                shots = self._shots(seg, ws, we)
                first = None
                eligible_i = 0
                for shot_i, (aname, t0, t1) in enumerate(shots):
                    ang = self.angles[aname]
                    subs = [sub for sub in self._subspans(ang, t0, t1) if sub[2] > 0]
                    # Subtle slow push-in (punch_in_scale over the whole shot) on
                    # every other long-enough shot, and on the calm tail — variety
                    # without flash. Skipped when a shot straddles a file boundary
                    # (the move would restart mid-shot).
                    is_tail = shot_i == len(shots) - 1 and len(shots) > 1
                    eligible = (self.p.punch_in and len(subs) == 1
                                and (t1 - t0) >= self.p.punch_in_min_seconds)
                    push = eligible and (is_tail or eligible_i % 2 == 1)
                    if eligible:
                        eligible_i += 1
                    for aid, mi, dur_f, ref_a, _ref_b in subs:
                        off = seg_start + (self.f(ref_a) - self.f(ws))
                        c = fcpxml.VideoClip(
                            ref=aid, name=aname, tl_off_f=off, dur_f=dur_f,
                            media_in_f=mi, src="video", mute=True,
                            punch_scale=self.p.punch_in_scale if push else 0.0)
                        items.append(c); seg_clips.append(c)
                        first = first or c
                        cursor = off + dur_f
                # single-source audio bed across [ws, we], anchored to first clip.
                # Each source angle may span multiple files; split at file boundaries
                # so no audio asset-clip requests frames past its file's declared end.
                bed_in = self.f(self.p.bed_fade_in_seconds) if si > 0 else 0
                bed_out = self.f(self.p.end_fade_seconds) if is_last else 0
                spans = self._bed_spans(seg, ws, we)
                flat = [(bname, aid, mi, dur_f, ref_a, ref_b, j)
                        for j, (bname, a, b) in enumerate(spans)
                        for aid, mi, dur_f, ref_a, ref_b in self._subspans(self.angles[bname], a, b)]
                for k, (bname, aid, mi, dur_f, ref_a, _ref_b, j) in enumerate(flat):
                    off = seg_start + (self.f(ref_a) - self.f(ws))
                    first.anchors.append(fcpxml.AudioClip(
                        ref=aid, name=f"{bname} audio (bed)", tl_off_f=off,
                        dur_f=dur_f, media_in_f=mi,
                        fade_in_f=bed_in if k == 0 else 0,
                        fade_out_f=bed_out if k == len(flat) - 1 else 0))

            # transition into the NEXT segment.  A centred cross dissolve borrows
            # D//2 of media from EACH side of the cut: the outgoing handle comes
            # from the tail we trim here, but the incoming clip must already have
            # D//2 of media before its in-point or FCP rejects the transition
            # ("Encountered an unexpected value").  Clamp to what's available; if
            # the incoming clip starts at its media head (no handle), hard cut.
            if D and not is_last:
                last = seg_clips[-1]
                head = self._head_handle_f(self.p.segments[si + 1])
                d_eff = min(D, 2 * head)
                d_eff -= d_eff % 2                     # keep d_eff//2 exact
                if d_eff >= 2 and last.dur_f > d_eff + 2:
                    last.dur_f -= d_eff
                    cursor -= d_eff
                    items.append(fcpxml.Dissolve(
                        tl_off_f=cursor - d_eff // 2, dur_f=d_eff))

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

    def _analysis_extents(self, names) -> dict[str, float]:
        """Local seconds (from each angle's t=0) that activity/content analysis
        must cover so every remix window the angle participates in lands inside
        the analysed range. Capped at the angle's own footage length; padded so
        boundary samples are never out-of-range."""
        margin = 5.0
        out: dict[str, float] = {}
        for s in self.p.segments:
            if s.type != "remix":
                continue
            _, we = self._window(s)
            for a in s.angles:
                ang = self.angles[a]
                local_end = we - ang.offset + margin
                out[a] = max(out.get(a, 0.0), local_end)
        for name in names:
            ang = self.angles[name]
            cap = ang.total_frames / self.fps
            out[name] = max(0.0, min(out.get(name, cap), cap))
        return out

    def _full_end(self, seg: Segment) -> float:
        return max(self.angles[a].offset + self.angles[a].total_frames / self.fps
                   for a in seg.angles)

    def _coverage_gaps(self, seg: Segment, ws: float, we: float,
                       margin: float = 0.0):
        """Sub-ranges of [ws, we] that NO angle's media spans at all. Uses the
        real media extent (margin 0), not the 0.4 s transition-safety margin
        `covers` applies — a window edge that merely lands within that safety
        buffer is handled gracefully during layout, not rejected here. A
        non-empty result means the window asks for time no camera filmed."""
        ivs = []
        for a in seg.angles:
            ang = self.angles[a]
            lo = max(ws, ang.offset + margin)
            hi = min(we, ang.offset + ang.total_frames / self.fps - margin)
            if hi > lo:
                ivs.append((lo, hi))
        ivs.sort()
        eps = 1.5 / self.fps        # ignore sub-frame gaps (rounding artifacts)
        gaps, cursor = [], ws
        for lo, hi in ivs:
            if lo > cursor + eps:
                gaps.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < we - eps:
            gaps.append((cursor, we))
        return gaps

    def _validate_coverage(self):
        """Fail early if any remix window has a stretch no angle covers, instead
        of silently collapsing that timeline span during layout."""
        problems = []
        for i, seg in enumerate(self.p.segments):
            if seg.type != "remix":
                continue
            ws, we = self._window(seg)
            gaps = self._coverage_gaps(seg, ws, we)
            if gaps:
                spans = ", ".join(f"{a:.1f}–{b:.1f}s" for a, b in gaps)
                problems.append(
                    f"remix segment {i} (window {ws:.1f}–{we:.1f}s, angles "
                    f"{seg.angles}): no angle covers {spans}")
        if problems:
            raise ValueError(
                "remix window(s) request time no angle covers — narrow the "
                "window, add a covering angle, or split it into a separate "
                "intro/remix segment:\n  - " + "\n  - ".join(problems))

    def _window(self, seg: Segment):
        return (seg.window if seg.window
                else [self.angles[seg.angle or seg.angles[0]].offset,
                      self._full_end(seg)])

    def _first_clip_media(self, seg: Segment):
        """(asset_id, media_in_f) of the first clip this segment will emit, or None.
        Mirrors how build() lays out each segment's opening clip."""
        if seg.type == "intro":
            ang = self.angles[seg.angle]
            return ang.asset_ids[0], ang.start_fs[0]
        if seg.type == "remix":
            ws, we = self._window(seg)
            for (aname, t0, t1) in self._shots(seg, ws, we):
                for aid, mi, dur_f, _ra, _rb in self._subspans(
                        self.angles[aname], t0, t1):
                    if dur_f > 0:
                        return aid, mi
        return None

    def _head_handle_f(self, seg: Segment) -> int:
        """Media frames available before a segment's first visible frame — how far
        a centred cross dissolve may reach back into the incoming clip."""
        fc = self._first_clip_media(seg)
        if not fc:
            return 0
        aid, mi = fc
        return max(0, mi - self.assets_by_id(aid).start_f)

    def _cut_points(self, ws: float, alt_end: float, pat: list[float]) -> list[float]:
        """Cut times across [ws, alt_end): the pattern's uneven lengths, with each
        interior cut slid to the strongest nearby audio onset (a rep landing, a
        beep) when cut_on_action is on — cuts land ON the action instead of on a
        metronome. A cut only snaps if it keeps at least 1.5 s of shot on both
        sides, so two cuts can never collapse onto one onset."""
        bounds, t, i = [ws], ws, 0
        while t < alt_end - 1e-6:
            t1 = min(t + pat[i % len(pat)], alt_end)
            if (self._ref_flux is not None and t1 < alt_end - 0.5):
                snapped = snap_to_onset(self._ref_flux, t1,
                                        self.p.cut_snap_seconds, self._onset_thr)
                if snapped - bounds[-1] >= 1.5 and snapped <= alt_end - 0.5:
                    t1 = snapped
            bounds.append(t1)
            t = t1
            i += 1
        return bounds

    def _activity_filter(self, covering: list[str], t0: float,
                         t1: float) -> list[str]:
        """Drop static/empty angles (motion below `_MIN_ACTIVITY`). A NaN reading
        (no data, or a window past the analysed range) means "no opinion" — the
        angle is kept, never excluded on missing data. Bypassed entirely when no
        covering angle clears the floor (e.g. a rest between rounds)."""
        acts = {a: self.angles[a].mean_activity(t0, t1) for a in covering}
        valid = [v for v in acts.values() if not math.isnan(v)]
        if valid and max(valid) >= self._MIN_ACTIVITY:
            kept = [a for a in covering
                    if math.isnan(acts[a]) or acts[a] >= self._MIN_ACTIVITY]
            return kept or covering
        return covering

    def _content_filter(self, active: list[str], t0: float,
                        t1: float) -> list[str]:
        """Among >1 candidates, keep those showing athletes (vision content).
        An unknown reading (NaN: no data or out-of-range window) is kept — we
        never drop an angle on missing data. Bypassed if it would empty the set,
        so we never drop to zero candidates."""
        if len(active) <= 1:
            return active
        with_athletes = [a for a in active
                         if math.isnan(c := self.angles[a].mean_content(t0, t1))
                         or c >= 0.5]
        return with_athletes or active

    # Score assigned to an unknown (NaN) activity/content reading: the midpoint
    # of the [0, 1] range, so missing data neither boosts a candidate to the top
    # nor sinks it to the bottom — it cannot beat a clearly-active angle, and is
    # not beaten by a clearly-weak one.
    _NEUTRAL = 0.5

    def _rank_pick(self, candidates: list[str], t0: float, t1: float,
                   last_pick: str | None, run_secs: float) -> str:
        """Best covering angle for [t0, t1] by weighted score (activity, content,
        coverage safety), minus a repeat penalty that grows with how long
        `last_pick` has already been held. Missing activity/content reads score
        neutrally, not as best-case. Deterministic: ties break to input order."""
        acts = {a: self.angles[a].mean_activity(t0, t1) for a in candidates}
        valid = [v for v in acts.values() if not math.isnan(v)]
        amax = max(valid) if valid else 0.0

        def score(a: str) -> float:
            ang = self.angles[a]
            act = acts[a]
            act_n = (self._NEUTRAL if (math.isnan(act) or amax <= 0)
                     else min(act / amax, 1.0))
            content = ang.mean_content(t0, t1)
            content_n = self._NEUTRAL if math.isnan(content) else content
            cover = ang.coverage_margin(t0, t1, self.fps)
            cover_n = max(0.0, min(cover / 2.0, 1.0))          # 2 s+ = full marks
            s = (self._W_ACTIVITY * act_n
                 + self._W_CONTENT * content_n
                 + self._W_COVERAGE * cover_n)
            if a == last_pick:
                s -= self._W_REPEAT * min(run_secs / self._REPEAT_FULL_S, 1.0)
            return s

        return max(candidates,
                   key=lambda a: (score(a), -candidates.index(a)))

    def _cover_tiles(self, pick: str, names: list[str], t0: float, t1: float):
        """Tile [t0, t1] with (angle, a, b) sub-shots so the whole span is
        emitted even when no single angle covers it (an angle hand-off mid-shot).

        The common case — one angle covers the whole shot — returns a single
        tile (pick, t0, t1). Where `pick` runs out, the span is continued by
        another covering angle, riding each angle to its media end before
        handing off. Sub-frame coverage gaps are absorbed into the adjacent tile
        (they vanish under frame rounding at emit time)."""
        eps = 0.5 / self.fps
        tiles: list[list] = []
        t = t0
        while t < t1 - eps:
            covering = [n for n in names
                        if self.angles[n].covers_media(t, min(t + eps, t1), self.fps)]
            if covering:
                a = pick if pick in covering else max(
                    covering, key=lambda n: self.angles[n].media_end(self.fps))
                b = min(t1, self.angles[a].media_end(self.fps))
                if b <= t:                       # safety: always make progress
                    b = min(t1, t + eps)
                tiles.append([a, t, b])
                t = b
            else:
                # sub-frame / no-coverage gap: absorb up to the next angle start
                # into the previous tile (clamped to real media when emitted).
                nxt = min((self.angles[n].offset for n in names
                           if self.angles[n].offset > t + eps), default=t1)
                nxt = min(nxt, t1)
                if tiles:
                    tiles[-1][2] = nxt
                else:
                    tiles.append([pick, t, nxt])
                t = nxt
        # merge adjacent same-angle tiles
        merged: list[list] = []
        for tl in tiles:
            if merged and merged[-1][0] == tl[0] and abs(merged[-1][2] - tl[1]) < eps:
                merged[-1][2] = tl[2]
            else:
                merged.append(tl)
        return [(a, a0, a1) for a, a0, a1 in merged] or [(pick, t0, t1)]

    def _shots(self, seg: Segment, ws: float, we: float):
        """Alternate angles with varied (onset-snapped) lengths; calm single-angle
        tail."""
        # last reference-time at which more than one angle still covers
        multi_end = we
        if seg.tail_single_angle and len(seg.angles) > 1:
            ends = sorted(self.angles[a].offset + self.angles[a].total_frames / self.fps
                          for a in seg.angles)
            multi_end = min(we, ends[-2] - 0.4)        # 2nd-latest angle runs out
        shots = []
        order_base = list(seg.angles)
        alt_end = min(we, multi_end)
        bounds = self._cut_points(ws, alt_end, seg.pattern)
        last_pick, run_secs = None, 0.0
        for t, t1 in zip(bounds, bounds[1:]):
            covering = [a for a in order_base if self.angles[a].covers(t, t1, self.fps)]
            if covering:
                # Gate out clearly-bad candidates (static/empty, then coach-only),
                # then rank the survivors by score rather than rotating to the
                # next angle, so a visibly better shot wins.
                active = self._activity_filter(covering, t, t1)
                active = self._content_filter(active, t, t1)
                pick = self._rank_pick(active, t, t1, last_pick, run_secs)
            else:
                # No single angle covers the whole shot (a hand-off falls inside
                # it; whole-window gaps are already rejected by
                # _validate_coverage). Seed with the angle covering the most of
                # it; _cover_tiles fills the rest from other covering angles.
                pick = max(order_base,
                           key=lambda a: self.angles[a].coverage_margin(t, t1, self.fps))
            # Split the shot across any angle hand-off so no frame is dropped.
            # Usually a single tile (one angle covers the whole shot).
            for tang, ta, tb in self._cover_tiles(pick, order_base, t, t1):
                run_secs = run_secs + (tb - ta) if tang == last_pick else (tb - ta)
                last_pick = tang
                shots.append((tang, ta, tb))
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
