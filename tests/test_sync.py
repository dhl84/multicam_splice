"""Sync offset cross-correlation + cut-on-action snapping (feedback issues 2, 9)."""
import unittest

import numpy as np

from multicam import sync


class TestXcorrOffset(unittest.TestCase):
    def _flux_with_peaks(self, n, idxs):
        f = np.zeros(n, dtype=np.float64)
        for i in idxs:
            f[i] = 10.0
        return f

    def test_offset_matches_event_lag(self):
        # ref event at index R, target event at index Q ->
        # offset (ref time of target t=0) = (R - Q) / FPS
        n = 4000
        R, Q = 2500, 2000
        ref = self._flux_with_peaks(n, [R])
        tgt = self._flux_with_peaks(n, [Q])
        off, psr = sync._xcorr(tgt, ref)
        self.assertAlmostEqual(off, (R - Q) / sync.FPS, places=3)
        self.assertGreater(psr, 8.0)

    def test_zero_offset_for_aligned_signals(self):
        n = 4000
        ref = self._flux_with_peaks(n, [1500, 2500])
        off, psr = sync._xcorr(ref.copy(), ref)
        self.assertAlmostEqual(off, 0.0, places=3)
        self.assertGreater(psr, 8.0)


class TestSpectralFluxStreaming(unittest.TestCase):
    """The chunked rewrite (issue 2) must match a single-shot FFT exactly."""

    def _mono(self, x):
        _SR, _HOP, _WIN = sync._SR, sync._HOP, sync._WIN
        n = (len(x) - _WIN) // _HOP
        if n <= 0:
            return np.zeros(0)
        idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
        frames = x[idx] * np.hanning(_WIN)
        logmag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
        flux = np.maximum(np.diff(logmag, axis=0, prepend=logmag[:1]), 0.0).sum(axis=1)
        med = np.convolve(flux, np.ones(200) / 200, mode="same")
        return np.maximum(flux - med, 0.0)

    def test_matches_monolithic_across_chunk_boundaries(self):
        rng = np.random.default_rng(1)
        orig = sync._stream_mono
        try:
            for L, ncuts in [(200000, 5), (333333, 7), (sync._WIN + sync._HOP, 2)]:
                x = rng.standard_normal(L).astype(np.float32)
                cuts = sorted(rng.choice(range(1, L), size=ncuts, replace=False))
                parts, prev = [], 0
                for c in list(cuts) + [L]:
                    parts.append(x[prev:c]); prev = c
                sync._stream_mono = lambda files, sr, _p=parts: iter(_p)
                got, exp = sync.spectral_flux(["x"]), self._mono(x)
                self.assertEqual(got.shape, exp.shape)
                if got.size:
                    self.assertTrue(np.allclose(got, exp, atol=1e-5, rtol=1e-4))
        finally:
            sync._stream_mono = orig


class TestSnapToOnset(unittest.TestCase):
    def setUp(self):
        # flux floor ~0 with one strong onset at t = 5.0 s
        self.flux = np.zeros(int(10 * sync.FPS))
        self.peak_i = int(5.0 * sync.FPS)
        self.flux[self.peak_i] = 50.0
        self.thr = sync.onset_threshold(self.flux)

    def test_snaps_to_nearby_onset(self):
        # proposed cut 0.3 s before the onset, window wide enough to reach it
        t = 5.0 - 0.3
        snapped = sync.snap_to_onset(self.flux, t, window_s=0.7, threshold=self.thr)
        self.assertAlmostEqual(snapped, self.peak_i / sync.FPS, places=2)

    def test_no_snap_when_onset_outside_window(self):
        t = 5.0 - 2.0   # 2 s away; window only 0.7 s
        snapped = sync.snap_to_onset(self.flux, t, window_s=0.7, threshold=self.thr)
        self.assertAlmostEqual(snapped, t, places=6)

    def test_no_snap_when_below_threshold(self):
        # raise the threshold above the peak: keep the metronome position
        t = 5.0 - 0.1
        snapped = sync.snap_to_onset(self.flux, t, window_s=0.7, threshold=1e9)
        self.assertAlmostEqual(snapped, t, places=6)

    def test_empty_flux_returns_t(self):
        self.assertEqual(sync.snap_to_onset(np.zeros(0), 3.0, 0.7, 1.0), 3.0)


if __name__ == "__main__":
    unittest.main()
