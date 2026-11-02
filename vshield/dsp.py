"""Lightweight audio DSP utilities (numpy/scipy only, no heavy ML deps).

Provides wav I/O, MFCCs, frame energy and an autocorrelation pitch tracker.
These power the *demo* backends. Production backends (NVIDIA TitaNet, AASIST /
Wav2Vec2 anti-spoofing) plug in through the same signal interfaces.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, rfft, irfft
from scipy.io import wavfile

SR = 16_000


def load_wav(path: str) -> tuple[np.ndarray, int]:
    sr, y = wavfile.read(path)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def save_wav(path: str, y: np.ndarray, sr: int = SR) -> None:
    y = np.clip(y, -1.0, 1.0)
    wavfile.write(path, sr, (y * 32767).astype(np.int16))


def telephone(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Narrowband phone channel: 8 kHz G.711 mu-law round trip, returned at `sr`."""
    from scipy.signal import resample_poly
    x = np.clip(resample_poly(y, 1, sr // 8000), -1.0, 1.0)
    mu = 255.0
    enc = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    enc = np.round((enc + 1) * 127.5) / 127.5 - 1          # 8-bit code words
    dec = np.sign(enc) * np.expm1(np.abs(enc) * np.log1p(mu)) / mu
    return resample_poly(dec, sr // 8000, 1).astype(np.float32)


def frame_signal(y: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    n = 1 + (len(y) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 60.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c > l:
            fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def mfcc(y: np.ndarray, sr: int = SR, n_mfcc: int = 20, n_fft: int = 512, n_mels: int = 40) -> np.ndarray:
    """Return MFCC matrix [frames, n_mfcc]."""
    frames = frame_signal(y, int(0.025 * sr), int(0.010 * sr)) * np.hamming(int(0.025 * sr))
    spec = np.abs(rfft(frames, n=n_fft)) ** 2
    mel = spec @ _mel_filterbank(sr, n_fft, n_mels).T
    return dct(np.log(mel + 1e-10), type=2, axis=1, norm="ortho")[:, :n_mfcc]


def frame_rms(y: np.ndarray, sr: int = SR, win: float = 0.025, hop: float = 0.010) -> np.ndarray:
    frames = frame_signal(y, int(win * sr), int(hop * sr))
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


def pitch_track(y: np.ndarray, sr: int = SR, fmin: float = 70.0, fmax: float = 400.0,
                win: float = 0.040, hop: float = 0.010):
    """Autocorrelation pitch tracker with parabolic interpolation.

    Returns (f0[frames], voicing_strength[frames], rms[frames]); f0 = 0 for unvoiced.
    """
    flen, h = int(win * sr), int(hop * sr)
    frames = frame_signal(y, flen, h)
    frames = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    w = frames * np.hanning(flen)
    n_fft = 2 * flen
    spec = rfft(w, n=n_fft)
    ac = irfft(np.abs(spec) ** 2, n=n_fft)[:, :flen]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(sr / fmax), int(sr / fmin)
    seg = ac[:, lo:hi]
    k = np.argmax(seg, axis=1)
    peak = seg[np.arange(len(seg)), k]
    lag = (k + lo).astype(float)
    # parabolic interpolation for sub-sample lag precision
    for i, L in enumerate(lag.astype(int)):
        if 1 <= L < flen - 1:
            a, b, c = ac[i, L - 1], ac[i, L], ac[i, L + 1]
            d = a - 2 * b + c
            if d != 0:
                lag[i] = L + 0.5 * (a - c) / d
    silence = rms < max(0.02 * rms.max(), 1e-4)
    voiced = (peak > 0.45) & ~silence
    f0 = np.where(voiced, sr / lag, 0.0)
    return f0, np.where(voiced, peak, 0.0), rms
