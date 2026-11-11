"""Signal B - Voiceprint match against the account holder's enrolled voice.

The similarity comes from NVIDIA TitaNet-Large (vshield/detectors.py): cosine similarity between the
caller's speaker embedding and the enrolled one, live or precomputed in the call file.

NOTE: a *match* does not prove a human - a good voice clone is built to match. That is exactly why
this signal is fused with the synthetic-voice signal.
"""
from __future__ import annotations

import numpy as np

from ..models import SignalResult

LABEL = "Voiceprint mismatch"
MIN_SPEECH_S = 2.0
# Mismatch at or below / match at or above. Calibrated on LibriSpeech dev-clean (40 speakers: same speaker
# >= 0.74, different speaker <= 0.62 even on a phone line) and checked on the held-out test-clean split.
THRESHOLDS = (0.55, 0.70)


def analyze(similarity: float | None, speech_s: float, account: dict) -> SignalResult:
    if similarity is None or speech_s < MIN_SPEECH_S:
        return SignalResult("voiceprint", LABEL, 0.5, 0.0, ["Waiting for 2 s of caller speech"], available=False)
    sim = float(similarity)
    lo, hi = THRESHOLDS
    score = float(np.clip(0.9 - 0.8 * (sim - lo) / (hi - lo), 0.1, 0.9))   # >= hi -> 0.1, <= lo -> 0.9
    matched = sim >= hi
    if matched:
        reasons = [f"Voice matches the enrolled voiceprint of {account['name']} (similarity {sim:.2f})"]
    elif sim <= lo:
        reasons = [f"Voice does NOT match the enrolled voiceprint of {account['name']} (similarity {sim:.2f})"]
    else:
        reasons = [f"Weak voiceprint match (similarity {sim:.2f})"]
    weight = min(1.0, speech_s / 6.0)
    return SignalResult("voiceprint", LABEL, score, weight, reasons,
                        {"similarity": round(sim, 3), "matched": bool(matched), "speech_seconds": round(speech_s, 1)})
