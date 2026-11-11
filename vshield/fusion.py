"""Risk fusion and action policy: all signals -> one explainable 1-99 % risk, verdict and action."""
from __future__ import annotations

import numpy as np

from .models import Decision, SignalResult

GAIN = 1.4           # sharpens the weighted-average log-odds
BIAS = -0.2          # slight prior toward "genuine" (most callers are real customers)
STEP_UP, ESCALATE = 40, 80
FULL_EVIDENCE = 3.5  # total signal weight at which the score is fully trusted

ACTIONS = {
    "PROCEED": "Proceed normally. Caller verified - no re-verification needed if transferred.",
    "STEP_UP": "Step-up verification: ask the suggested liveness challenge or push an OTP to the device on file. "
               "Hold credits, SIM/port and account changes until resolved.",
    "ESCALATE": "Do NOT apply credits, SIM swaps, port-outs or account changes. Warm-transfer to Fraud with the handoff card.",
    "AUTHORIZED_AGENT": "Caller disclosed an AI assistant. Send a consent request to the customer's phone on file "
                        "before discussing the account.",
}
LISTENING = "Low-risk call context. Answer normally - verification completes once the caller has spoken a few seconds."
HIGH_RISK_CHECK = ("High-risk change requested (SIM, port-out or new line): ask one liveness check before making it, "
                   "even when the voice looks genuine.")
CONSENTED = ("The customer approved this AI assistant on their phone. Share account information only; "
             "credits, SIM and account changes still need the customer.")


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1 - p)))


def fuse(signals: dict[str, SignalResult], *, high_risk_request: bool = False, liveness_passed: bool = False,
         consent_approved: bool = False) -> Decision:
    live = [s for s in signals.values() if s.available and s.weight > 0]
    if not live:
        return Decision(50.0, "Analysing...", "PROCEED", LISTENING, [])
    w = np.array([s.weight for s in live])
    z = np.array([_logit(s.score) for s in live])
    confidence = min(1.0, w.sum() / FULL_EVIDENCE)   # little evidence -> stay near the prior
    risk = 100 / (1 + np.exp(-(GAIN * confidence * (w @ z) / w.sum() + BIAS)))

    sv, vp, bh = signals.get("synthetic_voice"), signals.get("voiceprint"), signals.get("behavior")
    ch = signals.get("challenge")
    synthetic = sv is not None and sv.available and sv.score > 0.75 and sv.weight >= 0.8
    voice_match = vp is not None and vp.available and vp.details.get("matched")
    voice_heard = sv is not None and sv.available and vp is not None and vp.available
    pattern = None
    if synthetic and voice_match:
        pattern = "clone"
        risk = max(risk, 72.0)   # matching voice + synthetic speech = textbook clone
    elif synthetic and bh is not None and bh.available and bh.score > 0.65:
        pattern = "bot"
        risk = max(risk, 72.0)
    if ch is not None and ch.score > 0.5:
        risk = max(risk, 88.0)   # failed liveness challenge
    risk = float(np.clip(risk, 1, 99))   # never claim 0 % or 100 %

    if bh is not None and bh.details.get("ai_disclosed"):
        action = "AUTHORIZED_AGENT"
        verdict, text = (("AI assistant - customer consent confirmed", CONSENTED) if consent_approved
                         else ("Disclosed AI assistant", ACTIONS[action]))
    elif risk < STEP_UP:
        verdict = "Likely genuine customer" if voice_heard else "Low risk so far - listening"
        if high_risk_request and not liveness_passed:
            action, text = "STEP_UP", HIGH_RISK_CHECK
        else:
            action, text = "PROCEED", ACTIONS["PROCEED"] if voice_heard else LISTENING
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
        text = ACTIONS[action]

    # top reasons: the biggest contributors in the direction of the verdict
    contrib = []
    for s in live:
        c = s.weight * _logit(s.score)
        if (risk >= STEP_UP and c > 0) or (risk < STEP_UP and c < 0):
            contrib.append((abs(c), s.reasons[0] if s.reasons else s.label))
    if pattern == "clone":
        contrib.append((99, "Voice MATCHES the enrolled customer but is synthetic speech - a cloned voice"))
    reasons = [r for _, r in sorted(contrib, reverse=True)][:4]
    return Decision(round(risk, 1), verdict, action, text, reasons)
