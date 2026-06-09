"""Audio-based sync: place every angle on one common (reference) timeline.

The robust default is a spectral-flux onset function cross-correlated against the
reference, scored by peak-to-sidelobe ratio (PSR). Spectral flux captures the
sharp broadband transients shared across mics (impacts, claps, beeps), so it
locks even when the mics sound very different. A high PSR (say > 8) is a
confident lock; low PSR means the clips probably don't share audio (e.g. they
were filmed at non-overlapping times) and should be placed manually.

Each angle's files are treated as one contiguous recording.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 100              # onset-function frame rate
_SR = 22050           # decode rate for flux
_HOP = _SR // FPS
_WIN = 1024


@dataclass
class SyncResult:
    offset: float          # reference-time at which the angle's t=0 sits (s)
    psr: float             # peak-to-sidelobe ratio; higher = more confident
    confident: bool


def _decode_mono(path: Path, sr: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "a:0", "-ac", "1",
         "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _concat(files: list[Path], sr: int) -> np.ndarray:
    return np.concatenate([_decode_mono(Path(f), sr) for f in files])


def spectral_flux(files: list[Path]) -> np.ndarray:
    x = _concat(files, _SR)
    n = (len(x) - _WIN) // _HOP
    if n <= 0:
        return np.zeros(0)
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(_WIN)
    logmag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(np.diff(logmag, axis=0, prepend=logmag[:1]), 0.0).sum(axis=1)
    med = np.convolve(flux, np.ones(200) / 200, mode="same")
    return np.maximum(flux - med, 0.0)


def _xcorr(tmpl: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    t = tmpl - tmpl.mean()
    s = sig - sig.mean()
    nfft = 1 << int(np.ceil(np.log2(len(s) + len(t) - 1)))
    cc = np.fft.irfft(np.fft.rfft(s, nfft) * np.conj(np.fft.rfft(t, nfft)), nfft)
    cc = np.concatenate([cc[-(len(t) - 1):], cc[:len(s)]])
    lags = np.arange(-(len(t) - 1), len(s))
    peak = int(np.argmax(cc))
    mask = np.ones(len(cc), bool)
    mask[max(0, peak - FPS):peak + FPS] = False
    psr = (cc[peak] - cc[mask].mean()) / (cc[mask].std() + 1e-12)
    return lags[peak] / FPS, float(psr)


def sync_to_reference(ref_files: list[Path], target_files: list[Path],
                      min_psr: float = 8.0,
                      ref_flux: np.ndarray | None = None) -> SyncResult:
    """Offset (s) at which `target` starts within the `ref` timeline."""
    rf = ref_flux if ref_flux is not None else spectral_flux(ref_files)
    tf = spectral_flux(target_files)
    off, psr = _xcorr(tf, rf)
    return SyncResult(offset=off, psr=psr, confident=psr >= min_psr)
