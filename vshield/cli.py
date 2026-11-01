"""Command-line replay of the demo calls.

    python -m vshield.cli                 # all calls, summary
    python -m vshield.cli voice_clone -v  # one call, turn-by-turn + handoff card
"""
from __future__ import annotations

import argparse

from . import handoff
from .session import CallSession, list_calls, load_accounts, load_call

BAR = 30


def bar(risk: float) -> str:
    n = int(risk / 100 * BAR)
    return "#" * n + "." * (BAR - n)


def run(call_id: str, verbose: bool) -> None:
    call = load_call(call_id)
    sess = CallSession(call, load_accounts())
    print(f"\n=== {call['title']}  [ground truth: {call['ground_truth']}]")
    hist = sess.run()
    if verbose:
        for snap in hist:
            if snap.turn_idx < 0:
                who, text = "RING ", "(pre-answer: telecom + account context)"
            else:
                t = sess.turns_all[snap.turn_idx]
                who, text = ("AGENT" if t.speaker == "agent" else "CALL "), t.text
            print(f"{who} {bar(snap.risk)} {snap.risk:5.1f}%  {text[:70]}")
    final = hist[-1]
    print(f"\nFINAL  {final.risk:.0f}%  {final.verdict}  ->  {final.action}")
    for r in final.top_reasons:
        print(f"   - {r}")
    if verbose:
        print("\n" + handoff.handoff_card(call, sess.account, final, hist))
        if final.action == "ESCALATE":
            print("\n" + handoff.fraud_referral(call, sess.account, final))


def main() -> None:
    ap = argparse.ArgumentParser(description="Deepfake Voice Shield - replay demo calls")
    ap.add_argument("call", nargs="?", help=f"one of: {', '.join(list_calls())}")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    for cid in [a.call] if a.call else list_calls():
        run(cid, a.verbose or bool(a.call))


if __name__ == "__main__":
    main()
