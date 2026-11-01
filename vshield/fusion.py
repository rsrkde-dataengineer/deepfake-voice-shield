"""Risk fusion: combine all signals into one explainable 0-100 score + verdict + action."""
from __future__ import annotations

import numpy as np

from .models import SignalResult

GAIN = 1.4          # sharpens the weighted-average log-odds
BIAS = -0.2         # slight prior toward "genuine" (most callers are real customers)
STEP_UP, ESCALATE = 40, 75
FULL_EVIDENCE = 3.5  # total signal weight at which the score is fully trusted

ACTIONS = {
    "PROCEED": "Proceed normally. Caller verified - no re-verification needed if transferred.",
    "STEP_UP": "Step-up verification: ask the suggested liveness challenge or push an OTP to the device on file. "
               "Hold credits, SIM/port and account changes until resolved.",
    "ESCALATE": "Do NOT apply credits, SIM swaps, port-outs or account changes. Warm-transfer to Fraud with the handoff card.",
    "AUTHORIZED_AGENT": "Caller disclosed an AI assistant. Route to the authorised-agent flow (verify customer consent "
                        "on the device on file before discussing the account).",
}


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1 - p)))


def fuse(signals: dict[str, SignalResult]) -> tuple[float, str, str, list[str]]:
    live = [s for s in signals.values() if s.available and s.weight > 0]
    if not live:
        return 50.0, "Analysing...", "PROCEED", []
    w = np.array([s.weight for s in live])
    z = np.array([_logit(s.score) for s in live])
    confidence = min(1.0, w.sum() / FULL_EVIDENCE)   # little evidence -> stay near the prior
    risk = 100 / (1 + np.exp(-(GAIN * confidence * (w @ z) / w.sum() + BIAS)))

    ac, vp, bh = signals.get("acoustic"), signals.get("voiceprint"), signals.get("behavior")
    ch = signals.get("challenge")
    pattern = None
    synthetic = ac is not None and ac.available and ac.score > 0.75 and ac.weight >= 0.8
    voice_match = vp is not None and vp.available and vp.details.get("matched")
    if synthetic and voice_match:
        pattern = "clone"
        risk = max(risk, 72.0)   # matching voice + synthetic artefacts = textbook clone
    elif synthetic and bh is not None and bh.available and bh.score > 0.65:
        pattern = "bot"
        risk = max(risk, 72.0)
    if ch is not None and ch.score > 0.5:
        risk = max(risk, 88.0)
    risk = float(np.clip(risk, 1, 99))   # never claim 0 % or 100 %

    if bh is not None and bh.details.get("ai_disclosed"):
        verdict, action = "Disclosed AI assistant", "AUTHORIZED_AGENT"
    elif risk < STEP_UP:
        verdict, action = "Likely genuine customer", "PROCEED"
    else:
        if pattern == "clone":
            verdict = "Likely voice clone of the account holder"
        elif pattern == "bot" or (bh is not None and bh.score > 0.7):
            verdict = "Likely AI agent / bot"
        elif synthetic:
            verdict = "Likely synthetic voice"
        else:
            verdict = "Elevated risk - identity not confirmed"
        action = "STEP_UP" if risk < ESCALATE else "ESCALATE"

    # top reasons: biggest contributors in the direction of the verdict
    contrib = []
    for s in live:
        c = s.weight * _logit(s.score)
        if (risk >= STEP_UP and c > 0) or (risk < STEP_UP and c < 0):
            contrib.append((abs(c), s.reasons[0]))
    if pattern == "clone":
        contrib.append((99, "Voice MATCHES the enrolled customer but carries synthetic-speech artefacts - a cloned voice"))
    reasons = [r for _, r in sorted(contrib, reverse=True)][:4]
    return round(risk, 1), verdict, action, reasons
