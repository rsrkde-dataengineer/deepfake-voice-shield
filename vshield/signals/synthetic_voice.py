"""Signal A - Synthetic voice (TTS / voice clone) detection.

The probability comes from the Spectra-AASIST anti-spoofing model (vshield/detectors.py), either
live or precomputed in the call file. This module turns it into an explainable signal.
"""
from __future__ import annotations

from ..models import SignalResult

LABEL = "Synthetic voice detector"
MIN_SPEECH_S = 1.5


def analyze(probability: float | None, speech_s: float) -> SignalResult:
    if probability is None or speech_s < MIN_SPEECH_S:
        return SignalResult("synthetic_voice", LABEL, 0.5, 0.0, ["Waiting for caller speech"], available=False)
    p = float(probability)
    if p > 0.7:
        reasons = [f"Anti-spoofing model hears synthetic speech ({p:.0%} likely a TTS voice or voice clone)"]
    elif p < 0.3:
        reasons = [f"Anti-spoofing model hears a live human voice ({p:.0%} synthetic)"]
    else:
        reasons = [f"Anti-spoofing model is undecided ({p:.0%} synthetic)"]
    weight = min(1.3, 0.4 + speech_s / 6.0)   # trust grows with more speech
    return SignalResult("synthetic_voice", LABEL, p, weight, reasons,
                        {"synthetic_probability": round(p, 3), "speech_seconds": round(speech_s, 1)})
