"""Identity checks for a caller who sounds human but is not the account holder's enrolled voice.

A liveness challenge proves there is a live human on the line, which is no help against a human impersonator or a
bill-negotiation service calling as the customer. What they lack is the customer's history and the customer's phone:
  * an account-history question, matched against the account's own records ("historical record matching"). The
    records are things that are not printed on the bill, so a stolen or shared bill does not help;
  * a confirmation sent to the phone on file, which the real customer approves or rejects ("Not me").

The prototype matches answers with keywords from the record. In production an approved LLM or a fuzzy matcher can
judge the answer, and the questions rotate through the account's recent activity.
"""
from __future__ import annotations

import re

from .models import SignalResult, Turn

EVASIONS = ["don't know", "do not know", "not sure", "can't remember", "cannot remember", "forget", "forgot",
            "no idea", "would not venture", "can't say", "cannot say"]
PHONE_EVENTS = {"owner_approved": True, "owner_denied": False}   # the customer's tap on the phone on file
PHONE_CHECK = {"type": "phone_confirmation", "title": "Suggested identity check",
               "agent_script": "Send a confirmation request to the phone on file.",
               "why": "The real customer approves or taps 'Not me'. An impersonator does not have their phone."}


def _norm(text: str) -> str:
    return " " + " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()) + " "


def _record(account: dict, record_id: str) -> dict:
    return next(r for r in account.get("records", []) if r["id"] == record_id)


def suggest(account: dict, asked: set[str]) -> dict:
    """The next account-history question the caller has not been asked yet (or the phone check when none is left)."""
    rec = next((r for r in account.get("records", []) if r["id"] not in asked), None)
    if rec is None:
        return PHONE_CHECK
    return {"type": "record_question", "title": "Suggested identity question", "record": rec["id"],
            "agent_script": f'"One quick question from your account history: {rec["question"]}"',
            "why": f"Not printed on the bill, so a stolen or shared bill does not help. Record: {rec['answer']}.",
            "alternatives": [PHONE_CHECK]}


def evaluate(turns: list[Turn], account: dict) -> list[dict]:
    """Answers to account-history questions, and the customer's answer on their phone, in call order."""
    results = []
    for i, t in enumerate(turns):
        if t.speaker == "agent" and t.question and i + 1 < len(turns) and turns[i + 1].speaker == "caller":
            rec = _record(account, t.question["record"])
            answer = _norm(turns[i + 1].text)
            passed = any(_norm(k) in answer for k in rec["accept"])
            q, expected = rec["question"], rec["answer"]
            if passed:
                verdict, reason = "PASSED - matches the account record", f'Answered "{q}" correctly'
            elif any(e in answer for e in EVASIONS):
                verdict = f"FAILED - could not answer (record: {expected})"
                reason = f'Could not answer an account-history question: "{q}"'
            else:
                verdict = f"FAILED - does not match the account record ({expected})"
                reason = f'Answer to an account-history question does not match the record: "{q}"'
            results.append({"type": "record_question", "question": q, "answer": turns[i + 1].text,
                            "expected": expected, "passed": passed, "verdict": verdict, "reason": reason})
        if t.event in PHONE_EVENTS:
            passed = PHONE_EVENTS[t.event]
            results.append({"type": "phone_confirmation", "passed": passed,
                            "verdict": "PASSED - the account holder approved this call on their phone" if passed
                            else "FAILED - the account holder tapped 'Not me' on their phone",
                            "reason": "Account holder approved this call on the phone on file" if passed
                            else "Account holder tapped 'Not me' on the phone on file"})
    return results


def as_signal(results: list[dict]) -> SignalResult | None:
    if not results:
        return None
    denied = any(r["type"] == "phone_confirmation" and not r["passed"] for r in results)
    failed = [r for r in results if not r["passed"]]
    score = 0.97 if denied else 0.9 if failed else 0.08
    reasons = [r["reason"] for r in sorted(results, key=lambda r: (r["passed"], r["type"] == "record_question"))]
    return SignalResult("identity", "Identity check (account records)", score, 1.3, reasons,
                        {"results": results, "owner_denied": denied})
