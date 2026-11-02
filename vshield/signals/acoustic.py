"""Signal A - Synthetic voice (deepfake / TTS / voice-conversion) detection.

Backends (env VSHIELD_SPOOF_MODEL):
  demo           (default) voice-source micro-variation analysis (no ML deps).
  spectra        Spectra-AASIST (lab260, wav2vec2 XLS-R + AASIST), pinned revision
                 (pip install -r requirements-ml.txt).
  hf:<model_id>  any Hugging Face audio-classification anti-spoofing model, e.g. a
                 Wav2Vec2 / WavLM / AASIST-family detector (pip install transformers torch).

Why the demo backend works: human vocal folds never vibrate perfectly regularly
(micro-jitter of ~1-2 % cycle to cycle, constant pitch wander). Neural vocoders
produce unnaturally steady pitch periods. Real deployments should use a trained
anti-spoofing model (and fine-tune on 8 kHz telephony audio) - the demo backend is
a transparent stand-in so the end-to-end flow runs on any laptop.
"""
from __future__ import annotations

import hashlib
import importlib.util
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


# Spectra-AASIST: model.py is third-party code executed locally, so it is pinned to a reviewed revision.
SPECTRA_REPO = "lab260/Spectra-AASIST"
SPECTRA_REV = "eb65c2662d9e646d72557b3f4bdd08b000068c7f"
SPECTRA_WIN = 64600              # ~4 s crop the model was trained on
SPECTRA_THRESHOLD = -1.140625    # model card: bona fide logit below this -> spoof
SPECTRA_SCALE = 1.0              # logit distance -> probability steepness
_window_cache: dict[str, np.ndarray] = {}


@lru_cache(maxsize=1)
def _spectra():
    import torch
    from huggingface_hub import hf_hub_download
    from ..ml import device
    code = hf_hub_download(SPECTRA_REPO, "model.py", revision=SPECTRA_REV)
    spec = importlib.util.spec_from_file_location("spectra_aasist_model", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.SpectraAASIST.from_pretrained(SPECTRA_REPO, revision=SPECTRA_REV).eval().to(device())
    return model, torch


def _spectra_windows(y: np.ndarray) -> list[np.ndarray]:
    x = np.append(y[:1], y[1:] - 0.97 * y[:-1]).astype(np.float32)   # pre-emphasis, as in training
    if len(x) <= SPECTRA_WIN:
        return [np.resize(x, SPECTRA_WIN)]                             # repeat-pad short clips
    starts = list(range(0, len(x) - SPECTRA_WIN + 1, SPECTRA_WIN))
    if len(x) - (starts[-1] + SPECTRA_WIN) >= SR:                      # cover a tail of 1 s or more
        starts.append(len(x) - SPECTRA_WIN)
    return [x[s:s + SPECTRA_WIN] for s in starts]


def spectra_logits(y: np.ndarray) -> np.ndarray:
    """[windows, 2] logits (index 0 = spoof, 1 = bona fide). Windows already scored are cached,
    so re-scoring a growing call only runs the new audio."""
    model, torch = _spectra()
    wins = _spectra_windows(y)
    keys = [hashlib.sha1(w.tobytes()).hexdigest() for w in wins]
    todo = [(k, w) for k, w in zip(keys, wins) if k not in _window_cache]
    if todo:
        from ..ml import device
        with torch.inference_mode():
            out = model(torch.from_numpy(np.stack([w for _, w in todo])).to(device())).float().cpu().numpy()
        _window_cache.update({k: o for (k, _), o in zip(todo, out)})
    return np.stack([_window_cache[k] for k in keys])


def _spectra_score(y: np.ndarray) -> tuple[float, dict, list[str]]:
    logits = spectra_logits(y)
    bona = float(logits[:, 1].mean())
    p = float(1 / (1 + np.exp(-SPECTRA_SCALE * (SPECTRA_THRESHOLD - bona))))
    if p > 0.7:
        reasons = [f"Anti-spoofing model (Spectra-AASIST) detects synthetic speech - {p:.0%} likely TTS / voice clone"]
    elif p < 0.3:
        reasons = [f"Anti-spoofing model (Spectra-AASIST) finds no synthetic-speech artefacts ({p:.0%})"]
    else:
        reasons = [f"Anti-spoofing model (Spectra-AASIST) is undecided ({p:.0%} synthetic)"]
    return p, {"model": "spectra-aasist", "bonafide_logit": round(bona, 3), "windows": len(logits)}, reasons


@lru_cache(maxsize=1)
def _hf_pipeline(model_id: str):
    from transformers import pipeline  # optional heavy dependency
    from ..ml import device
    return pipeline("audio-classification", model=model_id, device=device())


def _hf_score(y: np.ndarray, model_id: str) -> tuple[float, dict, list[str]]:
    out = _hf_pipeline(model_id)({"raw": y.astype(np.float32), "sampling_rate": SR}, top_k=None)
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
    if BACKEND == "spectra":
        p, feats, reasons = _spectra_score(caller_audio)
    elif BACKEND.startswith("hf:"):
        p, feats, reasons = _hf_score(caller_audio, BACKEND[3:])
    else:
        p, feats, reasons = _demo_score(caller_audio)
    feats["speech_seconds"] = round(secs, 1)
    return SignalResult("acoustic", "Synthetic voice detector", p, min(1.3, 0.4 + secs / 6.0), reasons, feats)
