"""Generated documents: the transfer handoff card and (for high risk) a fraud referral.

This is the "Generated Document/Text" output in the brief. It travels with the call
on transfer so the next agent sees the fraud confidence instead of re-verifying
from scratch (the 60-90 s saving per transferred call).

Template-based so it runs offline. To use an LLM, pass the same facts dict to your
approved LLM with a "write a concise handoff note" prompt.
"""
from __future__ import annotations

from datetime import datetime

from .models import RiskSnapshot


def handoff_card(call: dict, account: dict, snap: RiskSnapshot, history: list[RiskSnapshot]) -> str:
    peak = max(h.risk for h in history) if history else snap.risk
    verified = snap.action == "PROCEED"
    t = call["telecom"]
    lines = [
        f"## Transfer handoff card - {call['call_id']}",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} by Deepfake Voice Shield_",
        "",
        f"**Account:** {call['account_id']} ({account['name']}) - {account['plan']}",
        f"**Caller ANI:** {t['ani']} (STIR/SHAKEN {t['stir_shaken']}, {t['line_type']})  |  **IVR intent:** {t['ivr_intent']}",
        "",
        f"### Fraud confidence: {snap.risk:.0f}% - {snap.verdict}",
        f"Peak risk during call: {peak:.0f}%",
        "",
        "**Why:**",
        *[f"- {r}" for r in snap.top_reasons],
        "",
    ]
    if snap.challenge_results:
        lines += ["**Liveness checks performed:**",
                  *[f"- Phrase repeat \"{r['phrase']}\": {r['verdict']}" for r in snap.challenge_results], ""]
    if verified:
        lines += [
            "### Receiving agent: NO re-verification needed",
            "- Voiceprint matched, live-human signals strong, verified caller ID.",
            "- Estimated time saved on this transfer: ~60-90 seconds.",
        ]
    else:
        lines += [
            "### Receiving agent: DO NOT treat this caller as verified",
            f"- Recommended action: {snap.action_text}",
            "- Do not repeat the same knowledge-based questions - a bot will simply answer them again.",
        ]
    return "\n".join(lines)


def fraud_referral(call: dict, account: dict, snap: RiskSnapshot) -> str:
    sig = snap.signals
    rows = "\n".join(f"| {s.label} | {s.score:.0%} | {s.reasons[0]} |" for s in sig.values() if s.available)
    asks = [t["text"] for t in call["turns"] if t["speaker"] == "caller"][:3]
    return "\n".join([
        f"## Fraud referral - {call['call_id']}",
        f"**Classification:** {snap.verdict} ({snap.risk:.0f}% confidence)",
        f"**Targeted account:** {call['account_id']} ({account['name']})",
        f"**Requested:** {call['telecom']['ivr_intent'].replace('_', ' ')}",
        f"**Origin:** {call['telecom']['ani']} via {call['telecom']['carrier']}",
        "",
        "| Signal | Fraud likelihood | Evidence |", "|---|---|---|", rows,
        "",
        "**Caller statements (first 3):**", *[f"> {a}" for a in asks],
        "",
        "**Suggested follow-up:** lock high-risk changes for 24h, notify account holder via device on file, "
        "add ANI to watch-list, attach audio to fraud case for model retraining.",
    ])
