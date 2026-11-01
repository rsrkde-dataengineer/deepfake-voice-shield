"""Signal A - Synthetic voice (deepfake / TTS / voice-conversion) detection.

Backends (env VSHIELD_SPOOF_MODEL):
  demo           (default) voice-source micro-variation analysis (no ML deps).
  hf:<model_id>  any Hugging Face audio-classification anti-spoofing model, e.g. a
                 Wav2Vec2 / WavLM / AASIST-family detector (pip install transformers torch).

Why the demo backend works: human vocal folds never vibrate perfectly regularly
(micro-jitter of ~1-2 % cycle to cycle, constant pitch wander). Neural vocoders
produce unnaturally steady pitch periods. Real deployments should use a trained
anti-spoofing model (and fine-tune on 8 kHz telephony audio) - the demo backend is
a transparent stand-in so the end-to-end flow runs on any laptop.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from ..dsp import SR, pitch_track
from ..models import SignalResult

BACKEND = os.getenv("VSHIELD_SPOOF_MODEL", "demo")
HUMAN_JITTER_FLOOR = 0.0025   # below this frame-to-frame pitch variation is suspicious


def voice_source_features(y: np.ndarray) -> dict:
    f0, peak, rms = pitch_track(y)
    v = f0 > 0
    pairs = v[1:] & v[:-1]
    if pairs.sum() < 10:
        return {}
    micro = np.abs(np.diff(f0))[pairs] / f0[1:][pairs]
    shim = np.abs(np.diff(rms))[pairs] / rms[1:][pairs]
    return {
        "micro_jitter": float(np.median(micro)),
        "shimmer": float(np.median(shim)),
        "periodicity": float(peak[v].mean()),
        "voiced_frames": int(v.sum()),
    }


def _demo_score(y: np.ndarray) -> tuple[float, dict, list[str]]:
    f = voice_source_features(y)
    if not f:
        return 0.5, f, ["Not enough voiced speech yet"]
    z = 4.0 * np.log10(HUMAN_JITTER_FLOOR / max(f["micro_jitter"], 1e-6))
    p = float(1 / (1 + np.exp(-z)))
    pct = f["micro_jitter"] * 100
    if p > 0.7:
        reasons = [f"Unnaturally steady vocal pitch (micro-jitter {pct:.2f}% vs human ~0.5-2%) - typical of TTS / voice cloning"]
    elif p < 0.3:
        reasons = [f"Natural vocal micro-variation (micro-jitter {pct:.2f}%) consistent with a live human voice"]
    else:
        reasons = [f"Borderline voice-source regularity (micro-jitter {pct:.2f}%)"]
    return p, f, reasons


@lru_cache(maxsize=1)
def _hf_pipeline(model_id: str):
    from transformers import pipeline  # optional heavy dependency
    return pipeline("audio-classification", model=model_id)


def _hf_score(y: np.ndarray, model_id: str) -> tuple[float, dict, list[str]]:
    out = _hf_pipeline(model_id)({"raw": y.astype(np.float32), "sampling_rate": SR})
    spoof_words, real_words = ("spoof", "fake", "synthetic", "deepfake", "ai"), ("bonafide", "bona", "real", "human", "genuine")
    p = None
    for o in out:
        lab = o["label"].lower()
        if any(w in lab for w in spoof_words):
            p = o["score"]
            break
        if any(w in lab for w in real_words):
            p = 1 - o["score"]
            break
    p = 0.5 if p is None else float(p)
    return p, {"model": model_id, "raw": out}, [f"Anti-spoofing model ({model_id}) synthetic probability {p:.0%}"]


def analyze(caller_audio: np.ndarray) -> SignalResult:
    secs = len(caller_audio) / SR
    if secs < 1.5:
        return SignalResult("acoustic", "Synthetic voice detector", 0.5, 0.0, ["Waiting for caller speech"],
                            available=False)
    if BACKEND.startswith("hf:"):
        p, feats, reasons = _hf_score(caller_audio, BACKEND[3:])
    else:
        p, feats, reasons = _demo_score(caller_audio)
    feats["speech_seconds"] = round(secs, 1)
    return SignalResult("acoustic", "Synthetic voice detector", p, min(1.3, 0.4 + secs / 6.0), reasons, feats)
