"""Signals D & E - Telecom metadata and account/security context.

Both are available the moment the call lands (before the agent says hello), so the
co-pilot shows a pre-answer risk. Rules are deliberately simple and explainable;
in production these become features in a trained model fed by CTI / ANI reputation
feeds and the account risk store.
"""
from __future__ import annotations

import numpy as np

from ..models import SignalResult

HIGH_RISK_INTENTS = {"sim_change": "SIM change", "port_out": "number port-out", "add_line": "add a line"}
CREDIT_INTENTS = {"cancel_service", "retention", "billing_dispute"}


def telecom(meta: dict, account: dict) -> SignalResult:
    s, reasons = 0.10, []
    att = meta.get("stir_shaken", "A")
    on_file = meta.get("ani") in account.get("anis_on_file", [])
    if att == "C":
        s += 0.25
        reasons.append("STIR/SHAKEN attestation C (caller ID not verified by originating carrier)")
        if on_file:
            s += 0.10
            reasons.append("ANI matches the account but with C attestation - possible caller-ID spoofing")
    elif att == "B":
        s += 0.12
        reasons.append("STIR/SHAKEN attestation B (partial)")
    if meta.get("line_type") == "voip":
        s += 0.15
        reasons.append(f"Originates from VoIP ({meta.get('carrier', 'unknown')})")
    if not on_file:
        s += 0.15
        reasons.append("Calling number is not on file for this account")
    if meta.get("calls_from_ani_24h", 0) > 5:
        s += 0.15
        reasons.append(f"{meta['calls_from_ani_24h']} calls from this number in 24h")
    if meta.get("accounts_called_from_ani_7d", 0) > 3:
        s += 0.20
        reasons.append(f"Same number touched {meta['accounts_called_from_ani_7d']} different accounts this week (bot-farm pattern)")
    dtmf = meta.get("ivr_dtmf_timing_ms") or []
    if len(dtmf) >= 3 and np.std(dtmf) < 30:
        s += 0.15
        reasons.append(f"IVR keypresses perfectly evenly spaced (~{int(np.mean(dtmf))} ms) - machine-entered")
    if not reasons:
        reasons.append("Known number, verified caller ID (STIR/SHAKEN A), mobile line")
    return SignalResult("telecom", "Telecom & ANI signals", float(np.clip(s, 0.05, 0.95)), 0.6, reasons,
                        {k: meta.get(k) for k in ("ani", "stir_shaken", "line_type", "carrier",
                                                  "calls_from_ani_24h", "accounts_called_from_ani_7d")})


def account_context(account: dict, meta: dict) -> SignalResult:
    s, reasons = 0.10, []
    intent = meta.get("ivr_intent", "")
    for flag in account.get("risk_flags", []):
        s += 0.20
        reasons.append(f"Account risk flag: {flag.replace('_', ' ')}")
    ev = account.get("security_events_7d", [])
    if ev:
        s += 0.15
        reasons.append("Recent security events: " + "; ".join(ev))
    if intent in HIGH_RISK_INTENTS:
        s += 0.20
        reasons.append(f"High-risk request ({HIGH_RISK_INTENTS[intent]}) - common account-takeover step")
    bh = account.get("billing_history", {})
    if intent in CREDIT_INTENTS and (bh.get("credits_90d", 0) >= 2 or bh.get("retention_offers_12m", 0) >= 2):
        s += 0.25
        reasons.append(f"Serial credit pattern: {bh.get('credits_90d', 0)} credits in 90 days, "
                       f"{bh.get('retention_offers_12m', 0)} retention offers in 12 months")
    if not reasons:
        reasons.append(f"Clean account history ({account.get('tenure_years', '?')}-year customer, no risk flags)")
    return SignalResult("account", "Account & security context", float(np.clip(s, 0.05, 0.95)), 0.6, reasons,
                        {"account": account.get("name"), "intent": intent, "plan": account.get("plan")})
