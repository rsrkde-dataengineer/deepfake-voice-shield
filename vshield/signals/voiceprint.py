"""Signal B - Voiceprint match against the account's enrolled voice.

Backends (env VSHIELD_VOICEPRINT):
  demo     (default) cepstral speaker embedding + background-model normalisation.
  titanet  NVIDIA TitaNet-Large via NeMo  (pip install "nemo_toolkit[asr]").

NOTE: a *match* does not prove a human - a good voice clone is designed to match.
That is exactly why this signal is fused with the anti-spoofing signal.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..dsp import SR, load_wav, mfcc, pitch_track
from ..models import SignalResult

DATA = Path(os.getenv("VSHIELD_DATA") or Path(__file__).resolve().parents[2] / "data").resolve()
BACKEND = os.getenv("VSHIELD_VOICEPRINT", "demo").lower()
TITANET = "nvidia/speakerverification_en_titanet_large"   # CC-BY-4.0, 192-dim, 16 kHz mono


def embed(y: np.ndarray, raw: bool = False) -> np.ndarray:
    """Demo speaker embedding: mean cepstrum over voiced frames + log-pitch."""
    M = mfcc(y)
    f0, _, _ = pitch_track(y)
    n = min(len(M), len(f0))
    voiced = f0[:n] > 0
    if voiced.sum() < 10:
        voiced = np.ones(n, bool)
    vec = np.concatenate([M[:n][voiced][:, 1:13].mean(0), [8 * np.log(np.median(f0[:n][voiced] + 1e-6) / 150)]])
    if raw:
        return vec
    ubm = _ubm()
    return (vec - ubm["mu"]) / ubm["sd"]


@lru_cache(maxsize=1)
def _ubm():
    return dict(np.load(DATA / "models" / "voiceprint_ubm.npz"))


@lru_cache(maxsize=1)
def _titanet():
    import nemo.collections.asr as nemo_asr  # heavy, optional
    from ..ml import device
    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(TITANET, map_location=device()).eval()


def _embed_titanet(y: np.ndarray) -> np.ndarray:
    emb, _ = _titanet().infer_segment(y.astype(np.float32))
    return emb.squeeze().detach().cpu().numpy()


def _embedding(y: np.ndarray) -> np.ndarray:
    return _embed_titanet(y) if BACKEND == "titanet" else embed(y)


@lru_cache(maxsize=16)
def enrolled_embedding(rel_path: str) -> np.ndarray:
    y, _ = load_wav(str(DATA / rel_path))
    return _embedding(y)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# Similarity thresholds (calibrate per backend on your own dev set). titanet: from scripts/evaluate.py on
# LibriSpeech dev-clean, 40 speakers - genuine min 0.74 clean / 0.72 phone, impostor max 0.51 / 0.62.
THRESH = {"demo": (0.55, 0.80), "titanet": (0.55, 0.70)}


def analyze(caller_audio: np.ndarray, account: dict) -> SignalResult:
    secs = len(caller_audio) / SR
    if secs < 2.0:
        return SignalResult("voiceprint", "Voiceprint mismatch", 0.5, 0.0, ["Waiting for 2 s of caller speech"],
                            available=False)
    sim = cosine(_embedding(caller_audio), enrolled_embedding(account["voiceprint"]))
    lo, hi = THRESH.get(BACKEND, THRESH["demo"])
    # map similarity to "mismatch" risk: >=hi -> 0.1, <=lo -> 0.9
    score = float(np.clip(0.9 - 0.8 * (sim - lo) / (hi - lo), 0.1, 0.9))
    matched = sim >= hi
    if matched:
        reasons = [f"Voice matches enrolled voiceprint of {account['name']} (similarity {sim:.2f})"]
    elif sim <= lo:
        reasons = [f"Voice does NOT match enrolled voiceprint of {account['name']} (similarity {sim:.2f})"]
    else:
        reasons = [f"Weak voiceprint match (similarity {sim:.2f})"]
    weight = min(1.0, secs / 6.0)  # trust grows with more speech
    return SignalResult("voiceprint", "Voiceprint mismatch", score, weight, reasons,
                        {"similarity": round(sim, 3), "matched": bool(matched), "backend": BACKEND,
                         "speech_seconds": round(secs, 1)})
