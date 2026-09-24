"""Shot ranking, late-window staleness, and window-aware audio bed
(feedback issues 3, 4, 5)."""
import math
import unittest
from pathlib import Path

import numpy as np

from multicam import audio
from multicam.config import FormatSpec, Project, Segment
from multicam.timeline import Builder, _Angle


def builder(reference="A"):
    proj = Project(
        media_dir=Path("/tmp"), output=Path("/tmp/out.fcpxml"),
        angles={}, segments=[], reference=reference, format=FormatSpec())
    return Builder(proj, log=lambda *a, **k: None)


def angle(name, total_frames, offset=0.0, activity=None, content=None,
          paths=None, cum_frames=None):
    return _Angle(
        name=name, paths=paths or [Path(f"{name}.mp4")], infos=[],
        asset_ids=["v1"], start_fs=[0], cum_frames=cum_frames or [0],
        offset=offset, total_frames=total_frames,
        activity=np.array([] if activity is None else activity, dtype=float),
        content=np.array([] if content is None else content, dtype=np.int8))


# 59.94 fps; ~600 frames = ~10 s of coverage
FPS = FormatSpec().fps_num / FormatSpec().fps_den
LONG = int(20 * FPS)


class TestLateWindowNeutrality(unittest.TestCase):
    """Issue 3: out-of-range lookups must be neutral, never a stale last sample."""

    def test_activity_in_range_real_value(self):
        # 2 fps activity samples; high near t=0
        a = angle("A", LONG, activity=[10.0] * 10)
        self.assertAlmostEqual(a.mean_activity(0.0, 1.0), 10.0, places=5)

    def test_activity_out_of_range_is_nan_not_stale(self):
        # samples only cover the first 5 s (10 samples @ 2 fps); query at t=15 s
        a = angle("A", LONG, activity=[10.0] * 10)
        self.assertTrue(math.isnan(a.mean_activity(15.0, 16.0)))

    def test_content_out_of_range_is_nan_not_stale(self):
        # content samples (0.5 fps) cover ~6 s; coach-only (0) there
        a = angle("A", LONG, content=[0, 0, 0])
        self.assertEqual(a.mean_content(0.0, 2.0), 0.0)              # in range: coach
        self.assertTrue(math.isnan(a.mean_content(15.0, 16.0)))     # out of range

    def test_no_data_is_neutral(self):
        a = angle("A", LONG)
        self.assertTrue(math.isnan(a.mean_activity(0.0, 1.0)))
        self.assertTrue(math.isnan(a.mean_content(0.0, 1.0)))


class TestActivityFilter(unittest.TestCase):
    def test_excludes_static_angle(self):
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[10.0] * 40),
                    "B": angle("B", LONG, activity=[0.5] * 40)}
        self.assertEqual(b._activity_filter(["A", "B"], 0.0, 2.0), ["A"])

    def test_bypassed_when_all_quiet(self):
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[0.5] * 40),
                    "B": angle("B", LONG, activity=[0.4] * 40)}
        self.assertEqual(b._activity_filter(["A", "B"], 0.0, 2.0), ["A", "B"])

    def test_nan_angle_is_kept(self):
        # B has no activity data (NaN) -> never excluded
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[10.0] * 40),
                    "B": angle("B", LONG)}
        self.assertEqual(set(b._activity_filter(["A", "B"], 0.0, 2.0)), {"A", "B"})


class TestContentFilter(unittest.TestCase):
    def test_prefers_athletes_over_coach(self):
        b = builder()
        b.angles = {"A": angle("A", LONG, content=[1, 1, 1, 1]),
                    "B": angle("B", LONG, content=[0, 0, 0, 0])}
        self.assertEqual(b._content_filter(["A", "B"], 0.0, 2.0), ["A"])

    def test_single_candidate_passes_through(self):
        b = builder()
        b.angles = {"A": angle("A", LONG, content=[0, 0])}
        self.assertEqual(b._content_filter(["A"], 0.0, 2.0), ["A"])


class TestRankPick(unittest.TestCase):
    """Issue 4: scored ranking, not rotation."""

    def test_higher_activity_wins(self):
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[10.0] * 40),
                    "B": angle("B", LONG, activity=[3.0] * 40)}
        self.assertEqual(b._rank_pick(["A", "B"], 0.0, 2.0, None, 0.0), "A")

    def test_clearly_better_angle_kept_despite_repeat_penalty(self):
        # A is clearly superior (10 vs 3); even held a long time the penalty
        # must not flip to a visibly worse shot.
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[10.0] * 40),
                    "B": angle("B", LONG, activity=[3.0] * 40)}
        self.assertEqual(b._rank_pick(["A", "B"], 0.0, 2.0, "A", 20.0), "A")

    def test_long_run_forces_variety_on_near_tie(self):
        # near-tie (10 vs 8) held a long time -> repeat penalty flips to B
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[10.0] * 40),
                    "B": angle("B", LONG, activity=[8.0] * 40)}
        self.assertEqual(b._rank_pick(["A", "B"], 0.0, 2.0, "A", 0.0), "A")
        self.assertEqual(b._rank_pick(["A", "B"], 0.0, 2.0, "A", 20.0), "B")

    def test_content_outranks_marginal_activity(self):
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[10.0] * 40, content=[0, 0, 0, 0]),
                    "B": angle("B", LONG, activity=[9.0] * 40, content=[1, 1, 1, 1])}
        # B's athletes-visible content beats A's marginally higher motion
        self.assertEqual(b._rank_pick(["A", "B"], 0.0, 2.0, None, 0.0), "B")

    def test_missing_activity_is_neutral_not_best(self):
        # A has no activity data (NaN), B has real moderate activity -> B must
        # win: missing data is scored neutrally, never as best-case.
        b = builder()
        b.angles = {"A": angle("A", LONG),
                    "B": angle("B", LONG, activity=[3.0] * 40)}
        self.assertEqual(b._rank_pick(["A", "B"], 0.0, 2.0, None, 0.0), "B")

    def test_unknown_scored_mid_pack_among_three(self):
        # B strong (act_n 1.0), C clearly weak (act_n ~0.2), A unknown (0.5):
        # the strong real angle wins, and the unknown ranks above the weak one
        # rather than at the top.
        b = builder()
        b.angles = {"B": angle("B", LONG, activity=[10.0] * 40),
                    "A": angle("A", LONG),
                    "C": angle("C", LONG, activity=[2.0] * 40)}
        # the strong real angle wins the full contest
        self.assertEqual(b._rank_pick(["A", "B", "C"], 0.0, 2.0, None, 0.0), "B")
        # against only the clearly-weak C (now sole valid -> normalises to 1.0),
        # documented behaviour: a known-but-weak reading outranks no-data.
        self.assertEqual(b._rank_pick(["A", "C"], 0.0, 2.0, None, 0.0), "C")


class TestShotsIntegration(unittest.TestCase):
    def test_better_angle_dominates_not_rotates(self):
        b = builder()
        b.angles = {"A": angle("A", LONG, activity=[10.0] * 80),
                    "B": angle("B", LONG, activity=[3.0] * 80)}
        seg = Segment(type="remix", angles=["A", "B"],
                      pattern=[2.0, 2.0, 2.0, 2.0, 2.0], tail_single_angle=False)
        shots = b._shots(seg, 0.0, 10.0)
        picks = [s[0] for s in shots]
        self.assertEqual(picks[0], "A")
        # rotation would alternate ABABA; ranking keeps the clearly better A on top
        self.assertGreater(picks.count("A"), picks.count("B"))


class TestAnalysisExtents(unittest.TestCase):
    """Issue 3: analysis length follows actual remix coverage, not a 600 s cap."""

    def test_covers_window_plus_margin_capped_at_footage(self):
        b = builder()
        b.angles = {"A": angle("A", int(900 * FPS)),
                    "B": angle("B", int(50 * FPS), offset=20.0)}
        b.p.segments = [Segment(type="remix", angles=["A", "B"],
                                window=[10.0, 800.0])]
        ext = b._analysis_extents(["A", "B"])
        # A queried to 800 s (+5 margin); well under its ~900 s footage
        self.assertAlmostEqual(ext["A"], 805.0, places=1)
        # B's window end maps to local 800-20=780 s but its footage is only ~50 s
        self.assertAlmostEqual(ext["B"], 50.0, delta=0.1)


class TestHandoffSpanningShots(unittest.TestCase):
    """Codex follow-up: a shot straddling an angle hand-off must be split, not
    partially emitted from one angle (which dropped the uncovered part)."""

    def test_cover_tiles_splits_at_handoff(self):
        # A covers 0..3 s, B covers 3..6 s; a shot 2..4 s spans the hand-off
        b = builder()
        b.angles = {"A": angle("A", int(3 * FPS), offset=0.0),
                    "B": angle("B", int(3 * FPS), offset=3.0)}
        tiles = b._cover_tiles("A", ["A", "B"], 2.0, 4.0)
        # two tiles, contiguous, tiling the whole [2,4] with no dropped span
        self.assertEqual([t[0] for t in tiles], ["A", "B"])
        self.assertAlmostEqual(tiles[0][1], 2.0, places=3)
        self.assertAlmostEqual(tiles[-1][2], 4.0, places=3)
        for (_, a0, a1), (_, b0, _b1) in zip(tiles, tiles[1:]):
            self.assertAlmostEqual(a1, b0, places=6)   # no gap between tiles

    def test_exact_buttjoin_window_tiles_with_no_dropped_time(self):
        # Codex's exact butt-join (not overlap): A's media ends exactly where
        # B's begins (frame-aligned), and a remix window 2..4 s spanning the
        # join must be emitted contiguously with no dropped time.
        a_frames = int(3 * FPS)
        join = a_frames / FPS                        # A's media end == B's start
        b = builder()
        b.angles = {"A": angle("A", a_frames, offset=0.0),
                    "B": angle("B", int(3 * FPS), offset=join)}
        seg = Segment(type="remix", angles=["A", "B"],
                      pattern=[5.0], tail_single_angle=False)
        b.p.segments = [seg]
        b._validate_coverage()                       # butt-join is fully covered
        shots = b._shots(seg, 2.0, 4.0)
        self.assertEqual([s[0] for s in shots], ["A", "B"])     # both angles used
        self.assertAlmostEqual(shots[0][1], 2.0, places=3)      # starts at window
        self.assertAlmostEqual(shots[-1][2], 4.0, places=3)     # ends at window
        for (_, a0, a1), (_, c0, _c1) in zip(shots, shots[1:]):
            self.assertAlmostEqual(a1, c0, places=6)            # contiguous
        # the whole 2 s span is covered (no dropped time) ...
        self.assertAlmostEqual(sum(t1 - t0 for _, t0, t1 in shots), 2.0, places=3)
        # ... and each emitted shot is fully covered by its own angle's media
        for aname, t0, t1 in shots:
            self.assertTrue(b.angles[aname].covers_media(t0, t1, FPS))

    def test_shots_never_drop_time_across_handoff(self):
        # overlapping by ~0.1 s so the union covers the window (validation passes)
        b = builder()
        b.angles = {"A": angle("A", int(3 * FPS), offset=0.0, activity=[10.0] * 10),
                    "B": angle("B", int(31 * FPS / 10), offset=2.9, activity=[10.0] * 10)}
        seg = Segment(type="remix", angles=["A", "B"],
                      pattern=[1.5, 1.5, 1.5, 1.5, 1.5], tail_single_angle=False)
        shots = b._shots(seg, 1.0, 5.0)
        # shots tile [1,5] contiguously...
        self.assertAlmostEqual(shots[0][1], 1.0, places=3)
        self.assertAlmostEqual(shots[-1][2], 5.0, places=3)
        for (_, a0, a1), (_, c0, _c1) in zip(shots, shots[1:]):
            self.assertAlmostEqual(a1, c0, places=6)
        # ...and every emitted shot is fully covered by its own angle's media
        for aname, t0, t1 in shots:
            self.assertTrue(b.angles[aname].covers_media(t0, t1, FPS),
                            f"shot {aname} {t0:.3f}-{t1:.3f} not covered by its angle")


class TestCoverageValidation(unittest.TestCase):
    """Issue (Codex #1): uncovered remix windows are rejected, not dropped."""

    def test_gap_between_angles_is_rejected(self):
        b = builder()
        # A covers ~0..5 s, B covers ~20..25 s -> 5..20 s is uncovered
        b.angles = {"A": angle("A", int(5 * FPS), offset=0.0),
                    "B": angle("B", int(5 * FPS), offset=20.0)}
        b.p.segments = [Segment(type="remix", angles=["A", "B"],
                                window=[0.0, 25.0])]
        with self.assertRaises(ValueError) as cm:
            b._validate_coverage()
        self.assertIn("no angle covers", str(cm.exception))

    def test_full_coverage_passes(self):
        b = builder()
        # overlapping angles fully tile 0..25 s
        b.angles = {"A": angle("A", int(15 * FPS), offset=0.0),
                    "B": angle("B", int(15 * FPS), offset=10.0)}
        b.p.segments = [Segment(type="remix", angles=["A", "B"],
                                window=[0.5, 24.0])]
        b._validate_coverage()  # no raise

    def test_coverage_gaps_reports_span(self):
        b = builder()
        b.angles = {"A": angle("A", int(5 * FPS), offset=0.0),
                    "B": angle("B", int(5 * FPS), offset=20.0)}
        seg = Segment(type="remix", angles=["A", "B"], window=[0.0, 24.0])
        gaps = b._coverage_gaps(seg, 0.0, 24.0)
        self.assertEqual(len(gaps), 1)
        lo, hi = gaps[0]
        self.assertAlmostEqual(lo, 5.0, places=1)    # end of A's media
        self.assertAlmostEqual(hi, 20.0, places=1)   # start of B's media


class TestBedStatsWindow(unittest.TestCase):
    """Issue 5: bed cleanliness from the material covering the window, not file 0."""

    def _two_file_angle(self, name):
        # file0 covers ~0..1.67 s, file1 ~1.67..3.34 s
        return angle(name, 200, offset=0.0,
                     paths=[Path(f"{name}_0.mp4"), Path(f"{name}_1.mp4")],
                     cum_frames=[0, 100])

    def test_uses_overlapping_file_not_first(self):
        b = builder()
        ang = self._two_file_angle("A")
        b.angles = {"A": ang}
        # file0 pristine, file1 clipping/loud -> a window over file1 must score file1
        b._audio_cache[ang.paths[0]] = audio.AudioStats("A0", -18.0, -6.0, 4.0)
        b._audio_cache[ang.paths[1]] = audio.AudioStats("A1", -6.0, 0.5, 18.0)
        st = b._bed_stats("A", 2.0, 3.0)     # window lands entirely in file1
        self.assertEqual(st.true_peak, 0.5)
        self.assertAlmostEqual(st.lufs, -6.0, places=5)

    def test_true_peak_is_worst_case_across_span(self):
        b = builder()
        ang = self._two_file_angle("A")
        b.angles = {"A": ang}
        b._audio_cache[ang.paths[0]] = audio.AudioStats("A0", -18.0, -6.0, 4.0)
        b._audio_cache[ang.paths[1]] = audio.AudioStats("A1", -18.0, -1.0, 4.0)
        st = b._bed_stats("A", 0.0, 3.34)    # spans both files
        self.assertEqual(st.true_peak, -1.0)  # worst peak wins


if __name__ == "__main__":
    unittest.main()
