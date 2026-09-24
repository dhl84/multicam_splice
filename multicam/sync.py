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

ACT_FPS = 2.0          # video activity samples per second


@dataclass
class SyncResult:
    offset: float          # reference-time at which the angle's t=0 sits (s)
    psr: float             # peak-to-sidelobe ratio; higher = more confident
    confident: bool


# Samples pulled from ffmpeg per read (~12 s at _SR). Bounds the live frame
# buffer so flux extraction never holds more than this many samples' worth of
# FFT frames at once, independent of total recording length.
_STREAM_BLOCK = 1 << 18


def _stream_mono(files: list[Path], sr: int):
    """Yield float32 mono blocks across all files as one continuous stream,
    decoding incrementally so the whole recording is never held in RAM."""
    for f in files:
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(Path(f)), "-map", "a:0",
             "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while True:
                raw = proc.stdout.read(_STREAM_BLOCK * 4)
                if not raw:
                    break
                yield np.frombuffer(raw, dtype=np.float32)
        finally:
            proc.stdout.close()
            if proc.wait() not in (0, None):
                raise subprocess.CalledProcessError(proc.returncode, "ffmpeg")


def spectral_flux(files: list[Path]) -> np.ndarray:
    """Spectral-flux onset function over all `files` treated as one recording.

    Streams the audio in blocks and reduces each to its (small) per-hop flux
    contribution, so peak memory is one block of FFT frames — not the whole
    decoded signal and frame-index matrix. The result is bit-for-bit equivalent
    to a single-shot FFT over the concatenated signal: blocks carry the tail
    samples needed for the next frame and the previous frame's spectrum for the
    cross-block difference, so frame alignment and the inter-frame diff are
    identical regardless of block boundaries."""
    win = np.hanning(_WIN)
    buf = np.zeros(0, dtype=np.float32)
    prev_logmag: np.ndarray | None = None
    flux_parts: list[np.ndarray] = []
    for block in _stream_mono(files, _SR):
        buf = block if buf.size == 0 else np.concatenate([buf, block])
        if len(buf) < _WIN:
            continue
        nfr = (len(buf) - _WIN) // _HOP + 1
        idx = np.arange(_WIN)[None, :] + _HOP * np.arange(nfr)[:, None]
        frames = buf[idx] * win
        logmag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
        prepend = (prev_logmag[None, :] if prev_logmag is not None
                   else logmag[:1])
        diff = np.diff(logmag, axis=0, prepend=prepend)
        flux_parts.append(np.maximum(diff, 0.0).sum(axis=1))
        prev_logmag = logmag[-1]
        buf = buf[nfr * _HOP:]                  # keep the next frame's lead-in
    if not flux_parts:
        return np.zeros(0)
    flux = np.concatenate(flux_parts)           # small: ~FPS samples/sec
    # The single-shot reference omits the final fitting frame (it uses
    # floor((L-WIN)/HOP) frames, not +1); drop our extra tail frame so the flux
    # array — and therefore every offset/PSR downstream — is identical.
    flux = flux[:-1]
    if len(flux) == 0:
        return np.zeros(0)
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


def onset_threshold(flux: np.ndarray, sigmas: float = 4.0) -> float:
    """A flux level that counts as a 'real' onset. The flux is mostly ~0 (it is
    median-subtracted) with sparse sharp spikes at true onsets, so mean + 4·std
    sits well above the noise floor and below genuine hits — cuts only snap to
    clearly strong events and otherwise keep their pattern position."""
    if len(flux) == 0:
        return float("inf")
    return float(flux.mean() + sigmas * flux.std())


def snap_to_onset(flux: np.ndarray, t: float, window_s: float,
                  threshold: float) -> float:
    """Slide a proposed cut time t (seconds on the flux timeline) to the
    strongest audio onset within ±window_s — a rep landing, a beep, a shout —
    so the cut lands ON the action. Returns t unchanged when no onset in the
    window clears `threshold` (keep the metronome rather than snap to noise)."""
    if len(flux) == 0 or window_s <= 0:
        return t
    i = round(t * FPS)
    w = round(window_s * FPS)
    lo, hi = max(0, i - w), min(len(flux), i + w + 1)
    if hi <= lo:
        return t
    win = flux[lo:hi]
    peak = int(np.argmax(win))
    if win[peak] < threshold:
        return t
    return (lo + peak) / FPS


def video_activity(files: list[Path], max_seconds: float | None = None) -> np.ndarray:
    """Per-sample visual motion energy at ACT_FPS samples/sec (higher = more movement).

    Decodes each file to a tiny greyscale raster (64×36) and returns the
    mean absolute frame-difference signal — the same idea as spectral_flux but
    for picture.  Near-zero values mean the camera is pointing at a static or
    empty scene; high values mean people / action are visible.
    """
    parts = []
    remaining = max_seconds
    for p in files:
        if remaining is not None and remaining <= 0:
            break
        cmd = ["ffmpeg", "-v", "error", "-i", str(p)]
        if remaining is not None:
            cmd += ["-t", f"{remaining:.3f}"]
        cmd += ["-vf", f"scale=64:36,fps={ACT_FPS}",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        n = len(raw) // (64 * 36)
        if n:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, 64 * 36).astype(np.float32)
            parts.append(arr)
            if remaining is not None:
                remaining -= n / ACT_FPS
    if not parts:
        return np.zeros(0)
    frames = np.concatenate(parts)
    return np.abs(np.diff(frames, axis=0, prepend=frames[:1])).mean(axis=1)


def sync_to_reference(ref_files: list[Path], target_files: list[Path],
                      min_psr: float = 8.0,
                      ref_flux: np.ndarray | None = None) -> SyncResult:
    """Offset (s) at which `target` starts within the `ref` timeline."""
    rf = ref_flux if ref_flux is not None else spectral_flux(ref_files)
    tf = spectral_flux(target_files)
    off, psr = _xcorr(tf, rf)
    return SyncResult(offset=off, psr=psr, confident=psr >= min_psr)
