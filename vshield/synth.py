"""Synthetic speech generator for the demo dataset (100% fake, no real voices).

A classic source-filter model: a glottal pulse train shaped by vowel formant
resonators. Two "renderers" mimic the acoustic difference the anti-spoofing
signal looks for:

* human  - natural micro-jitter, shimmer, pitch wander, breath + room noise
* tts    - the regularity typical of neural TTS / voice-conversion vocoders:
           near-zero jitter, smooth pitch, perfectly clean background

Speaker identity (pitch + vocal-tract length + spectral tilt) is separate from
the renderer, so a "clone" can share a customer's identity yet be rendered as TTS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .dsp import SR

VOWELS = {
    "a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
    "e": (530, 1840, 2480), "o": (570, 840, 2410), "ae": (660, 1720, 2410),
}


@dataclass(frozen=True)
class Speaker:
    name: str
    f0: float            # mean pitch (Hz)
    tract: float         # formant scale (shorter vocal tract -> higher formants)
    tilt: float          # glottal spectral tilt (0.90 bright .. 0.98 dark)
    vowel_bias: tuple = (1.0, 1.0, 1.0)  # personal formant colouring


@dataclass(frozen=True)
class Renderer:
    jitter: float        # per-period pitch randomness (fraction)
    shimmer: float       # per-period amplitude randomness (fraction)
    wander: float        # slow pitch random walk (fraction)
    breath: float        # aspiration noise level (relative)
    room: float          # background noise floor (absolute)
    syl_var: float       # syllable duration variability


HUMAN = Renderer(jitter=0.018, shimmer=0.12, wander=0.035, breath=0.10, room=0.004, syl_var=0.45)
TTS = Renderer(jitter=0.0015, shimmer=0.01, wander=0.004, breath=0.06, room=0.004, syl_var=0.08)


def _resonator(F: float, B: float):
    r = np.exp(-np.pi * B / SR)
    a = [1.0, -2 * r * np.cos(2 * np.pi * F / SR), r * r]
    b = [1.0 - r]
    return b, a


def _syllable(spk: Speaker, rnd: Renderer, dur: float, vowel: str, f0_base: float, rng) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n)
    t, f0 = 0, f0_base
    while t < n:
        f0 *= 1 + rnd.wander * 0.15 * rng.standard_normal()  # slow drift
        f0 = float(np.clip(f0, 0.7 * f0_base, 1.3 * f0_base))
        period = SR / (f0 * (1 + rnd.jitter * rng.standard_normal()))
        out[t] = 1.0 + rnd.shimmer * rng.standard_normal()
        t += max(int(round(period)), 1)
    exc = lfilter([1.0], [1.0, -spk.tilt], out)
    if rnd.breath > 0:
        exc += rnd.breath * np.std(exc) * rng.standard_normal(n)
    y = exc
    for F, B, bias in zip(VOWELS[vowel], (70, 100, 140), spk.vowel_bias):
        b, a = _resonator(F * spk.tract * bias, B)
        y = lfilter(b, a, y)
    att = min(int(0.025 * SR), n // 3)
    env = np.ones(n)
    env[:att] = np.sin(np.linspace(0, np.pi / 2, att)) ** 2
    env[-att:] = env[:att][::-1]
    y = y * env
    return y / (np.max(np.abs(y)) + 1e-9)


def say(text: str, spk: Speaker, rnd: Renderer, seed: int = 0) -> np.ndarray:
    """Render `text` as pseudo-speech (prosody/rhythm follows word & syllable count)."""
    rng = np.random.default_rng(seed)
    vowels = list(VOWELS)
    words = text.split()
    chunks = [np.zeros(int(0.05 * SR))]
    n_words = len(words)
    k = int(rng.integers(len(vowels)))  # vowels cycle evenly -> phonetic balance
    for wi, w in enumerate(words):
        n_syl = max(1, sum(ch in "aeiouy" for ch in w.lower()) // 1)
        n_syl = min(n_syl, 4)
        decl = 1.0 + 0.08 * (1 - wi / max(n_words - 1, 1))  # declination
        for s in range(n_syl):
            dur = 0.16 * (1 + rnd.syl_var * (rng.random() - 0.5))
            v = vowels[k % len(vowels)]
            k += 1
            f0 = spk.f0 * decl * (1 + 0.06 * np.sin(wi * 1.3 + s))
            chunks.append(0.6 * _syllable(spk, rnd, dur, v, f0, rng))
            chunks.append(np.zeros(int(0.015 * SR)))
        gap = 0.07 * (1 + rnd.syl_var * 1.5 * (rng.random() - 0.3))
        if w.endswith((",", ".", "?", "-", "—")):
            gap += 0.18
        chunks.append(np.zeros(max(int(gap * SR), 1)))
    y = np.concatenate(chunks)
    y = y + rnd.room * rng.standard_normal(len(y))
    return (0.7 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)
