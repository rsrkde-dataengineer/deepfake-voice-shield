"""Streaming call session: feed turns one at a time, get a risk snapshot after each.

In production `add_turn` is driven by streaming ASR + VAD over the live audio; here
we replay the mock call files turn by turn, which exercises exactly the same logic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import challenge, fusion
from .dsp import SR, load_wav
from .models import RiskSnapshot, Turn
from .signals import acoustic, behavior, context, voiceprint

DATA = Path(os.getenv("VSHIELD_DATA") or Path(__file__).resolve().parents[1] / "data").resolve()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No call data in {DATA}. For the real-audio calls, unzip the shared demo data "
                                "there or build it with scripts/build_sample_set.py (see README).")
    return path


def load_accounts() -> dict:
    return json.loads(_require(DATA / "accounts.json").read_text())


def list_calls() -> list[str]:
    return sorted(p.stem for p in _require(DATA / "calls").glob("*.json"))


def load_call(call_id: str) -> dict:
    return json.loads((DATA / "calls" / f"{call_id}.json").read_text())


class CallSession:
    def __init__(self, call: dict, accounts: dict | None = None):
        self.call = call
        self.account = (accounts or load_accounts())[call["account_id"]]
        self.turns_all = [Turn.from_dict(t) for t in call["turns"]]
        self.audio, _ = load_wav(str(DATA / call["caller_audio"]))
        self.turns: list[Turn] = []
        self.history: list[RiskSnapshot] = []
        # context signals are known before the call is answered
        self._telecom = context.telecom(call["telecom"], self.account)
        self._account = context.account_context(self.account, call["telecom"])

    @property
    def done(self) -> bool:
        return len(self.turns) >= len(self.turns_all)

    def _caller_audio_so_far(self) -> np.ndarray:
        spans = [t.audio_span for t in self.turns if t.speaker == "caller" and t.audio_span]
        if not spans:
            return np.zeros(0, dtype=np.float32)
        return self.audio[: int(spans[-1][1] * SR)]

    def snapshot(self) -> RiskSnapshot:
        y = self._caller_audio_so_far()
        sig = {
            "acoustic": acoustic.analyze(y),
            "voiceprint": voiceprint.analyze(y, self.account),
            "behavior": behavior.analyze(self.turns),
            "telecom": self._telecom,
            "account": self._account,
        }
        ch_results = challenge.evaluate(self.turns, behavior._caller_latencies(self.turns))
        ch_sig = challenge.as_signal(ch_results)
        if ch_sig:
            sig["challenge"] = ch_sig
        risk, verdict, action, reasons = fusion.fuse(sig)
        issued = sum(1 for t in self.turns if t.challenge)
        pending = bool(self.turns) and bool(self.turns[-1].challenge)   # asked, not yet answered
        suggested = None
        if action in ("STEP_UP", "ESCALATE") and not ch_results and not pending:
            suggested = challenge.suggest(self.call["call_id"], issued)
        snap = RiskSnapshot(turn_idx=len(self.turns) - 1, risk=risk, verdict=verdict, action=action,
                            action_text=fusion.ACTIONS[action], signals=sig, top_reasons=reasons,
                            suggested_challenge=suggested, challenge_results=ch_results)
        self.history.append(snap)
        return snap

    def start(self) -> RiskSnapshot:
        """Pre-answer snapshot (telecom + account context only)."""
        return self.snapshot()

    def add_turn(self) -> RiskSnapshot:
        self.turns.append(self.turns_all[len(self.turns)])
        return self.snapshot()

    def run(self) -> list[RiskSnapshot]:
        self.start()
        while not self.done:
            self.add_turn()
        return self.history
