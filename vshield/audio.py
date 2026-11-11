"""Audio helpers: WAV I/O, resampling, a simulated phone line and a quality check.

numpy/scipy only, so the dashboard runs without any ML packages.
"""
from __future__ import annotations

import io

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, welch

SR = 16_000            # what the models analyse (16 kHz mono)
PLAYBACK_SR = 24_000   # full-call recordings, for listening


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Mono float32 in [-1, 1] and the sample rate."""
    sr, y = wavfile.read(path)
    y = y.astype(np.float32) / 32768.0 if y.dtype == np.int16 else y.astype(np.float32)
    return (y.mean(axis=1) if y.ndim > 1 else y), sr


def save_wav(path: str, y: np.ndarray, sr: int = SR) -> None:
    wavfile.write(path, sr, (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16))


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, sr, (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16))
    return buf.getvalue()


def resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y.astype(np.float32)
    g = np.gcd(sr_from, sr_to)
    return resample_poly(y, sr_to // g, sr_from // g).astype(np.float32)


def telephone(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Narrowband phone line: 8 kHz G.711 mu-law round trip, returned at `sr`."""
    x = np.clip(resample(y, sr, 8000), -1.0, 1.0)
    mu = 255.0
    enc = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    enc = np.round((enc + 1) * 127.5) / 127.5 - 1          # 8-bit code words
    dec = np.sign(enc) * np.expm1(np.abs(enc) * np.log1p(mu)) / mu
    return resample(dec, 8000, sr)


def quality(y: np.ndarray, sr: int) -> dict:
    """Numbers that catch broken audio: clipping, silence, and muffled 'buzz' with no consonants."""
    f, p = welch(y, sr, nperseg=1024)
    frame = int(0.02 * sr)
    n = len(y) // frame
    rms = np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    active = rms > 10 ** (-45 / 20)
    return {
        "seconds": round(len(y) / sr, 2),
        "peak": round(float(np.abs(y).max()), 3),
        "clipped_share": float(np.mean(np.abs(y) > 0.99)),
        "speech_share": round(float(active.mean()), 3),
        "speech_level_db": round(float(20 * np.log10(np.sqrt((rms[active] ** 2).mean()))) if active.any() else -120, 1),
        "energy_above_2khz": round(float(p[f > 2000].sum() / (p.sum() + 1e-12)), 4),
    }
