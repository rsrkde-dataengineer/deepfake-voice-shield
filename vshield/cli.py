"""Command-line replay of the demo calls.

    python -m vshield.cli                    # all calls, summary
    python -m vshield.cli voice_clone        # one call: turn by turn + handoff card + fraud referral
    python -m vshield.cli --live             # re-run the models live (needs requirements-ml.txt)
"""
from __future__ import annotations

import argparse

from . import handoff
from .session import CallSession, list_calls, load_accounts, load_call

BAR = 30


def bar(risk: float) -> str:
    n = int(risk / 100 * BAR)
    return "#" * n + "." * (BAR - n)


def run(call_id: str, verbose: bool, live: bool) -> None:
    call = load_call(call_id)
    sess = CallSession(call, load_accounts(), live=live)
    print(f"\n=== {call['title']}  [ground truth: {call['ground_truth']}]")
    hist = sess.run()
    if verbose:
        for snap in hist:
            if snap.turn_idx < 0:
                who, text = "RING ", "(before answer: telecom + account context)"
            else:
                t = sess.turns_all[snap.turn_idx]
                who, text = ("AGENT" if t.speaker == "agent" else "CALL "), t.text
            print(f"{who} {bar(snap.risk)} {snap.risk:5.1f}%  {snap.action:16s} {text[:60]}")
    final = hist[-1]
    print(f"FINAL  {final.risk:.0f}%  {final.verdict}  ->  {final.action}"
          f"   (risk path: {' > '.join(f'{h.risk:.0f}' for h in hist)})")
    for r in final.top_reasons:
        print(f"   - {r}")
    if verbose:
        print("\n" + handoff.handoff_card(call, sess.account, final, hist))
        if final.action in ("ESCALATE", "STEP_UP"):
            print("\n" + handoff.fraud_referral(call, sess.account, final))


def main() -> None:
    ap = argparse.ArgumentParser(description="Deepfake Voice Shield - replay the demo calls")
    ap.add_argument("call", nargs="?", help="call id (default: all calls)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--live", action="store_true", help="run Spectra-AASIST and TitaNet live instead of using stored scores")
    a = ap.parse_args()
    for cid in [a.call] if a.call else list_calls():
        run(cid, a.verbose or bool(a.call), a.live)


if __name__ == "__main__":
    main()
