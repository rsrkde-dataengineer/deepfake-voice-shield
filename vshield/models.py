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
    audio_span: Optional[tuple[float, float]] = None   # caller audio slice (seconds in caller wav)
    challenge: Optional[dict] = None                    # agent-issued liveness challenge
    interrupts: bool = False                            # agent barged in on the caller

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        span = d.get("audio_span")
        return cls(idx=d["idx"], speaker=d["speaker"], text=d["text"], start=d["start"], end=d["end"],
                   audio_span=tuple(span) if span else None, challenge=d.get("challenge"),
                   interrupts=d.get("interrupts", False))


@dataclass
class SignalResult:
    name: str                    # machine name, e.g. "acoustic"
    label: str                   # display name
    score: float                 # 0 = looks genuine/human, 1 = looks fraudulent/synthetic
    weight: float                # how much fusion trusts this signal right now (0..1.5)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    available: bool = True


@dataclass
class RiskSnapshot:
    turn_idx: int
    risk: float                  # 0..100
    verdict: str                 # "Likely genuine customer" / "Likely voice clone" / ...
    action: str                  # PROCEED / STEP_UP / ESCALATE
    action_text: str
    signals: dict[str, SignalResult]
    top_reasons: list[str]
    suggested_challenge: Optional[dict] = None
    challenge_results: list[dict] = field(default_factory=list)
