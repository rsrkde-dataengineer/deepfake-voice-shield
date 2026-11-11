"""Generated documents: the transfer handoff card and, for risky calls, a fraud referral.

This is the "Generated Document/Text" output in the brief. The handoff card travels with the call on
transfer, so the next agent sees the fraud confidence instead of re-verifying from scratch (the 60-90 s
saving per transferred call). Template-based so it runs offline; an approved LLM can be given the same
facts to write it in free text.
"""
from __future__ import annotations

from datetime import datetime

from .models import RiskSnapshot


def is_verified(snap: RiskSnapshot) -> bool:
    return snap.action == "PROCEED" and snap.verdict == "Likely genuine customer"


def _evidence(snap: RiskSnapshot) -> list[str]:
    out = []
    sv, vp = snap.signals.get("synthetic_voice"), snap.signals.get("voiceprint")
    if sv is not None and sv.available:
        out.append(f"- Synthetic-voice probability (Spectra-AASIST): {sv.score:.0%}")
    if vp is not None and vp.available:
        out.append(f"- Voiceprint similarity (NVIDIA TitaNet): {vp.details['similarity']:.2f}"
                   f" ({'match' if vp.details['matched'] else 'no match'})")
    return out


def handoff_card(call: dict, account: dict, snap: RiskSnapshot, history: list[RiskSnapshot]) -> str:
    peak = max(h.risk for h in history) if history else snap.risk
    t = call["telecom"]
    lines = [
        f"## Transfer handoff card - {call['call_id']}",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} by Deepfake Voice Shield_",
        "",
        f"**Account:** {call['account_id']} ({account['name']}) - {account['plan']}  ",
        f"**Caller ID:** {t['ani']} (STIR/SHAKEN {t['stir_shaken']}, {t['line_type']})  ",
        f"**Reason for call:** {t['ivr_intent'].replace('_', ' ')}",
        "",
        f"### {snap.verdict} - fraud risk {snap.risk:.0f}% (peak {peak:.0f}%)",
        "",
        "**Why:**",
        *[f"- {r}" for r in snap.top_reasons],
        *_evidence(snap),
        "",
    ]
    if snap.challenge_results:
        lines += ["**Liveness checks done:**",
                  *[f"- Repeat \"{r['phrase']}\": {r['verdict']}" for r in snap.challenge_results], ""]
    if is_verified(snap):
        lines += ["### Receiving agent: NO re-verification needed",
                  "- Voiceprint matched, live human voice, liveness and caller-ID checks done on this call.",
                  "- Estimated time saved on this transfer: 60-90 seconds."]
    elif snap.action == "AUTHORIZED_AGENT":
        lines += ["### Receiving agent: caller is an AI assistant acting for the customer",
                  f"- {snap.action_text}"]
    else:
        lines += ["### Receiving agent: do NOT treat this caller as verified",
                  f"- Recommended action: {snap.action_text}",
                  "- Do not repeat the same knowledge questions - a bot with breached data simply answers them again."]
    return "\n".join(lines)


def fraud_referral(call: dict, account: dict, snap: RiskSnapshot) -> str:
    rows = "\n".join(f"| {s.label} | {s.score:.0%} | {s.reasons[0]} |" for s in snap.signals.values() if s.available)
    asks = [t["text"] for t in call["turns"] if t["speaker"] == "caller"][:3]
    t = call["telecom"]
    return "\n".join([
        f"## Fraud referral - {call['call_id']}",
        f"**Classification:** {snap.verdict} ({snap.risk:.0f}% risk)  ",
        f"**Targeted account:** {call['account_id']} ({account['name']})  ",
        f"**Requested:** {t['ivr_intent'].replace('_', ' ')}  ",
        f"**Origin:** {t['ani']} via {t['carrier']} (STIR/SHAKEN {t['stir_shaken']})",
        "",
        "| Signal | Fraud likelihood | Evidence |", "|---|---|---|", rows,
        "",
        "**Caller statements (first 3):**", *[f"> {a}" for a in asks],
        "",
        "**Suggested follow-up:** hold high-risk changes for 24 h, notify the account holder on the device on file, "
        "add the calling number to the watch-list, and attach the audio to the fraud case for model retraining.",
    ])
