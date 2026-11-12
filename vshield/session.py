"""Call session: feed turns one at a time, get a risk snapshot after each.

In production `add_turn` is driven by streaming ASR + voice activity detection on the live call;
here the demo calls are replayed turn by turn, which exercises exactly the same logic.

Model scores come from the call file (precomputed by Spectra-AASIST and TitaNet with
scripts/score_calls.py), or from the models themselves with live=True (needs requirements-ml.txt).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import challenge, fusion, identity
from .audio import SR, load_wav
from .models import RiskSnapshot, Turn
from .signals import behavior, context, synthetic_voice, voiceprint

DATA = Path(os.getenv("VSHIELD_DATA") or Path(__file__).resolve().parents[1] / "data").resolve()
STORY = ["genuine_customer", "genuine_new_number", "voice_clone", "human_impersonator", "ai_negotiator",
         "human_negotiator", "ai_assistant"]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No call data in {DATA}. Build it with scripts/build_demo_calls.py "
                                "and scripts/score_calls.py (see README).")
    return path


def load_accounts() -> dict:
    return json.loads(_require(DATA / "accounts.json").read_text())


def list_calls() -> list[str]:
    ids = [p.stem for p in _require(DATA / "calls").glob("*.json")]
    return sorted(ids, key=lambda c: (STORY.index(c) if c in STORY else len(STORY), c))


def load_call(call_id: str) -> dict:
    return json.loads(_require(DATA / "calls" / f"{call_id}.json").read_text())


class CallSession:
    def __init__(self, call: dict, accounts: dict | None = None, live: bool = False):
        self.call = call
        self.account = (accounts or load_accounts())[call["account_id"]]
        self.turns_all = [Turn.from_dict(t) for t in call["turns"]]
        self.turns: list[Turn] = []
        self.history: list[RiskSnapshot] = []
        self.live = live
        if live:
            self.audio, _ = load_wav(str(DATA / call["audio"]["caller"]))
            self.enrolled, _ = load_wav(str(DATA / self.account["voiceprint"]))
        # telecom and account context are known before the call is answered
        self._telecom = context.telecom(call["telecom"], self.account)
        self._account = context.account_context(self.account, call["telecom"])

    @property
    def done(self) -> bool:
        return len(self.turns) >= len(self.turns_all)

    def model_scores(self) -> tuple[float | None, float | None, float]:
        """(synthetic-voice probability, voiceprint similarity, seconds of caller speech so far)."""
        caller = [t for t in self.turns if t.speaker == "caller" and t.audio_span]
        if not caller:
            return None, None, 0.0
        secs = caller[-1].audio_span[1]
        if not self.live:
            s = caller[-1].scores or {}
            return s.get("synthetic_voice"), s.get("voice_similarity"), secs
        from . import detectors
        y = self.audio[: int(secs * SR)]
        if secs < synthetic_voice.MIN_SPEECH_S:
            return None, None, secs
        return detectors.synthetic_probability(y), detectors.voice_similarity(y, self.enrolled), secs

    def snapshot(self) -> RiskSnapshot:
        p_synth, sim, secs = self.model_scores()
        sig = {
            "synthetic_voice": synthetic_voice.analyze(p_synth, secs),
            "voiceprint": voiceprint.analyze(sim, secs, self.account),
            "behavior": behavior.analyze(self.turns),
            "telecom": self._telecom,
            "account": self._account,
        }
        results = challenge.evaluate(self.turns, behavior.caller_latencies(self.turns))
        ch_sig = challenge.as_signal(results)
        if ch_sig:
            sig["challenge"] = ch_sig
        checks = identity.evaluate(self.turns, self.account)
        id_sig = identity.as_signal(checks)
        if id_sig:
            sig["identity"] = id_sig
        d = fusion.fuse(sig, high_risk_request=self.call["telecom"].get("ivr_intent") in context.HIGH_RISK_INTENTS,
                        liveness_passed=any(r["passed"] for r in results),
                        consent_approved=any(t.event == "consent_approved" for t in self.turns))
        last = self.turns[-1] if self.turns else None
        pending = last is not None and bool(last.challenge or last.question)   # asked, not yet answered
        suggested = None
        if d.action in ("STEP_UP", "ESCALATE") and not pending:
            if d.pattern == "other_voice":     # a human who is not the enrolled voice: liveness proves nothing here
                if not checks:
                    suggested = identity.suggest(self.account, {t.question["record"] for t in self.turns if t.question})
            elif not results:
                suggested = challenge.suggest(self.call, sum(1 for t in self.turns if t.challenge))
        snap = RiskSnapshot(turn_idx=len(self.turns) - 1, risk=d.risk, verdict=d.verdict, action=d.action,
                            action_text=d.action_text, signals=sig, top_reasons=d.reasons,
                            suggested_challenge=suggested, challenge_results=results, identity_results=checks)
        self.history.append(snap)
        return snap

    def start(self) -> RiskSnapshot:
        """Pre-answer snapshot (telecom + account context only)."""
        return self.snapshot()

    def add_turn(self) -> RiskSnapshot:
        if self.done:
            return self.history[-1]
        self.turns.append(self.turns_all[len(self.turns)])
        return self.snapshot()

    def run(self) -> list[RiskSnapshot]:
        self.start()
        while not self.done:
            self.add_turn()
        return self.history


def call_audio(call: dict) -> tuple[np.ndarray, int]:
    """The whole call (agent + caller) for listening."""
    return load_wav(str(DATA / call["audio"]["call"]))
