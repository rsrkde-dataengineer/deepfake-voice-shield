"""Signal C - Conversational behaviour (works on the live transcript + timestamps).

Catches AI agents even when their voice is good:
  * response-latency regularity   (bots answer with near-constant delay)
  * disfluency rate               (humans say um/uh, restart sentences)
  * barge-in reaction             (humans stop when interrupted; many bots talk over)
  * negotiation-bot playbook      (scripted retention/credit phrasing)
  * social-engineering cues       (OTP avoidance, urgency, redirecting codes)
  * AI self-disclosure            (routes to an authorised-agent flow instead of fraud)

In production the transcript comes from streaming ASR (e.g. NVIDIA Parakeet / Riva or
Whisper) and an LLM can replace the phrase lists below; the rule version keeps the
prototype explainable and dependency-free.
"""
from __future__ import annotations

import re

import numpy as np

from ..models import SignalResult, Turn

FILLERS = r"\b(um+|uh+|erm|hmm|you know|i mean|sorry|oh|wait|let me think|yeah)\b"
BOT_PLAYBOOK = [
    "loyal customer", "competitor", "comparable plan", "retention", "port my number", "recurring monthly credit",
    "monthly credit", "waive", "activation fees", "for twelve months", "confirm the credit", "otherwise i will",
]
SOCIAL_ENG = [
    "can't access", "cannot access", "different email", "send the code to", "instead", "urgent", "hurry",
    "right away", "stolen", "new sim", "today",
]
# Only an explicit AI disclosure: a human saying "on behalf of" is a third party, not an AI assistant.
AI_DISCLOSURE = ["i am an ai", "i'm an ai", "an ai assistant", "automated assistant", "virtual assistant calling"]


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1 - p)))


def caller_latencies(turns: list[Turn]) -> list[float]:
    lats = []
    for i, t in enumerate(turns):
        if t.speaker != "caller" or i == 0:
            continue
        prev = turns[i - 1]
        if prev.speaker != "agent" or prev.interrupts or prev.challenge:
            continue  # skip barge-ins and challenge answers (scored separately)
        if i >= 2 and turns[i - 2].speaker == "caller" and turns[i - 2].end > prev.end:
            continue  # caller was still talking over the agent
        lats.append(round(t.start - prev.end, 2))
    return lats


def analyze(turns: list[Turn]) -> SignalResult:
    caller = [t for t in turns if t.speaker == "caller"]
    if not caller:
        return SignalResult("behavior", "Conversation behaviour", 0.5, 0.0, ["Waiting for caller to speak"],
                            available=False)
    text = " ".join(t.text for t in caller).lower()
    n_words = max(len(text.split()), 1)
    subs: list[tuple[float, float, str]] = []   # (score, weight, reason)
    details: dict = {}

    # 1. latency regularity
    lats = caller_latencies(turns)
    details["latencies_s"] = lats
    if len(lats) >= 2:
        cv = float(np.std(lats) / (np.mean(lats) + 1e-6))
        details["latency_cv"] = round(cv, 3)
        s = float(np.clip(0.9 - (cv - 0.05) * 3.0, 0.08, 0.92))
        why = (f"Response delay nearly constant ({np.mean(lats):.2f}s +/- {np.std(lats):.2f}s) - machine-like"
               if s > 0.6 else f"Natural, variable response timing ({min(lats):.1f}-{max(lats):.1f}s)")
        subs.append((s, 1.0, why))

    # 2. disfluencies
    fill = len(re.findall(FILLERS, text)) + text.count("—")
    rate = 100 * fill / n_words
    details["disfluencies_per_100w"] = round(rate, 1)
    if n_words >= 15:
        s = float(np.clip(0.8 - rate * 0.09, 0.1, 0.8))
        why = (f"Zero hesitations or self-corrections in {n_words} words - unusually polished speech"
               if fill == 0 else f"{fill} natural hesitations/self-corrections ({rate:.0f} per 100 words)")
        subs.append((s, 0.8 if n_words < 40 else 1.0, why))

    # 3. barge-in reaction
    for i, t in enumerate(turns):
        if t.speaker == "agent" and t.interrupts and i > 0 and turns[i - 1].speaker == "caller":
            overrun = turns[i - 1].end - t.start
            details["barge_in_overrun_s"] = round(overrun, 2)
            if overrun > 0.8:
                subs.append((0.9, 1.3, f"Kept talking {overrun:.1f}s over the agent's interruption - typical of a voice bot"))
            else:
                subs.append((0.12, 1.0, "Stopped promptly when interrupted (natural turn-taking)"))

    # 4. negotiation-bot playbook
    hits = [p for p in BOT_PLAYBOOK if p in text]
    details["playbook_phrases"] = hits
    if len(hits) >= 2:
        subs.append((min(0.55 + 0.08 * len(hits), 0.92), 1.0,
                     f"Scripted negotiation playbook detected ({len(hits)} phrases: {', '.join(hits[:3])}...)"))

    # 5. social engineering cues
    se = [p for p in SOCIAL_ENG if p in text]
    details["social_engineering_cues"] = se
    if len(se) >= 3:
        subs.append((min(0.5 + 0.07 * len(se), 0.88), 0.9,
                     f"Social-engineering pattern: {', '.join(se[:4])}"))

    disclosed = any(p in text for p in AI_DISCLOSURE)
    details["ai_disclosed"] = disclosed

    if not subs:
        return SignalResult("behavior", "Conversation behaviour", 0.5, 0.2, ["Gathering conversational evidence"],
                            details)
    w = np.array([s[1] for s in subs])
    z = np.array([_logit(s[0]) for s in subs])
    score = float(1 / (1 + np.exp(-(w @ z) / w.sum())))
    reasons = [s[2] for s in sorted(subs, key=lambda s: -abs(s[0] - 0.5) * s[1])]
    weight = min(1.2, 0.3 + 0.25 * len(subs))
    return SignalResult("behavior", "Conversation behaviour", score, weight, reasons, details)
