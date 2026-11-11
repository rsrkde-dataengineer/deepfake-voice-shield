"""Active liveness challenges.

When passive signals land in the grey zone the co-pilot suggests a challenge the
agent can ask naturally. A random phrase forces a bot / voice-clone pipeline to do
fresh ASR -> LLM -> TTS generation, which shows up as a latency spike relative to
the caller's own baseline. (Humans just repeat it.) Answer content is checked too.

Challenge design notes for the pitch:
  * "What's your zip?" is weak - a bot with breached data answers instantly.
  * Random-phrase repeat, an off-script question about the caller's surroundings,
    and an OTP push to the device on file are much harder to automate.
"""
from __future__ import annotations

import re

import numpy as np

from .models import SignalResult, Turn

PHRASES = ["blue river forty seven", "green lantern sixty two", "silver maple eighteen", "orange harbor ninety one"]
OTHER_CHALLENGES = [
    {"type": "context_question", "prompt": "Quick question while that loads - what's the weather like where you are right now?"},
    {"type": "otp_device", "prompt": "Send a push confirmation to the device on file (not email/SMS to a new number)."},
]


def suggest(call: dict, already_issued: int) -> dict:
    """A random phrase to repeat. Demo calls carry the phrase their agent actually used."""
    phrase = call.get("challenge_phrase") or PHRASES[(sum(map(ord, call["call_id"])) + already_issued) % len(PHRASES)]
    return {
        "type": "phrase_repeat", "phrase": phrase,
        "agent_script": f'"Before we continue, could you please repeat this phrase for me: {phrase}."',
        "why": "Random phrases force a bot/clone to generate new audio on the fly - watch the response delay.",
        "alternatives": OTHER_CHALLENGES,
    }


def _norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def evaluate(turns: list[Turn], baseline_latencies: list[float]) -> list[dict]:
    results = []
    base = float(np.median(baseline_latencies)) if baseline_latencies else 0.9
    for i, t in enumerate(turns):
        if t.speaker != "agent" or not t.challenge or i + 1 >= len(turns):
            continue
        ans = turns[i + 1]
        if ans.speaker != "caller":
            continue
        latency = ans.start - t.end
        ratio = latency / max(base, 0.2)
        exact = _norm(t.challenge.get("phrase", "")) == _norm(ans.text)
        slow = latency > max(1.8, 2.0 * base)
        results.append({
            "type": t.challenge["type"], "phrase": t.challenge.get("phrase"), "answer": ans.text,
            "latency_s": round(latency, 2), "baseline_s": round(base, 2), "ratio": round(ratio, 1),
            "content_ok": exact, "passed": exact and not slow,
            "verdict": ("FAILED - response delay jumped %.1fx over caller's baseline (%.1fs vs %.1fs)" % (ratio, latency, base))
            if slow else ("PASSED - natural, prompt repetition" if exact else "FAILED - phrase not repeated correctly"),
        })
    return results


def as_signal(results: list[dict]) -> SignalResult | None:
    if not results:
        return None
    failed = [r for r in results if not r["passed"]]
    score = 0.92 if failed else 0.1
    return SignalResult("challenge", "Liveness challenge failure", score, 1.3,
                        [f"Liveness challenge {r['verdict']}" for r in results], {"results": results})
