"""Core data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Turn:
    idx: int
    speaker: str                 # "agent" | "caller"
    text: str
    start: float                 # seconds from call start
    end: float
    audio_span: Optional[tuple[float, float]] = None   # caller audio slice (seconds in the caller-only track)
    challenge: Optional[dict] = None                    # agent-issued liveness challenge
    question: Optional[dict] = None                     # agent-asked identity question, e.g. {"record": "device"}
    interrupts: bool = False                            # agent barged in on the caller
    event: Optional[str] = None                         # customer's answer on their phone: consent_approved, owner_denied
    scores: Optional[dict] = None                       # precomputed model scores for the caller audio so far
    source: Optional[str] = None                        # where the caller audio comes from

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        span = d.get("audio_span")
        return cls(idx=d["idx"], speaker=d["speaker"], text=d["text"], start=d["start"], end=d["end"],
                   audio_span=tuple(span) if span else None, challenge=d.get("challenge"), question=d.get("question"),
                   interrupts=d.get("interrupts", False), event=d.get("event"), scores=d.get("scores"),
                   source=d.get("source"))


@dataclass
class SignalResult:
    name: str                    # machine name, e.g. "synthetic_voice"
    label: str                   # display name
    score: float                 # 0 = looks genuine/human, 1 = looks fraudulent/synthetic
    weight: float                # how much fusion trusts this signal right now (0..1.5)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    available: bool = True


@dataclass
class Decision:
    risk: float                  # 1..99 (never 0 or 100)
    verdict: str                 # "Likely genuine customer" / "Likely voice clone of the account holder" / ...
    action: str                  # PROCEED / STEP_UP / ESCALATE / AUTHORIZED_AGENT
    action_text: str
    reasons: list[str]
    pattern: Optional[str] = None   # "clone", "bot" or "other_voice" (a live human who is not the enrolled voice)


@dataclass
class RiskSnapshot:
    turn_idx: int                # -1 = before the call is answered
    risk: float
    verdict: str
    action: str
    action_text: str
    signals: dict[str, SignalResult]
    top_reasons: list[str]
    suggested_challenge: Optional[dict] = None          # the check the agent should ask next (liveness or identity)
    challenge_results: list[dict] = field(default_factory=list)   # liveness challenges
    identity_results: list[dict] = field(default_factory=list)    # account-history questions, phone confirmations
