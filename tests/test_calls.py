"""The five demo calls, end to end, on the stored model scores (no ML packages needed)."""
import pytest

from vshield import handoff
from vshield.session import CallSession, list_calls, load_accounts, load_call

EXPECTED = {
    "genuine_customer": ("PROCEED", "Likely genuine customer"),
    "genuine_new_number": ("PROCEED", "Likely genuine customer"),
    "voice_clone": ("ESCALATE", "Likely voice clone of the account holder"),
    "ai_negotiator": ("ESCALATE", "Likely AI agent / bot"),
    "ai_assistant": ("AUTHORIZED_AGENT", "AI assistant - customer consent confirmed"),
}


@pytest.fixture(scope="module")
def runs():
    accounts = load_accounts()
    out = {}
    for cid in list_calls():
        sess = CallSession(load_call(cid), accounts)
        out[cid] = (sess, sess.run())
    return out


def test_all_use_cases_present():
    assert list_calls() == list(EXPECTED)


@pytest.mark.parametrize("cid", list(EXPECTED))
def test_final_decision(runs, cid):
    sess, hist = runs[cid]
    action, verdict = EXPECTED[cid]
    assert (hist[-1].action, hist[-1].verdict) == (action, verdict)
    assert sess.call["expected_action"] == action


def test_never_claims_certainty(runs):
    assert all(1 <= h.risk <= 99 for _, hist in runs.values() for h in hist)


def test_no_verified_claim_before_the_caller_speaks(runs):
    ring = runs["genuine_customer"][1][0]
    assert ring.action == "PROCEED" and ring.verdict != "Likely genuine customer"
    assert not handoff.is_verified(ring)


def test_clone_fools_the_voiceprint_but_not_shield(runs):
    sess, hist = runs["voice_clone"]
    final = hist[-1]
    assert final.signals["voiceprint"].details["matched"], "a clone should fool a voiceprint-only check"
    assert final.signals["synthetic_voice"].score > 0.75
    before = [h for h in hist if not h.challenge_results and h.turn_idx >= 1]
    assert all(h.action == "STEP_UP" for h in before), "steps up first, so the challenge can confirm the clone"
    assert any(h.suggested_challenge and h.suggested_challenge["phrase"] == "silver maple eighteen" for h in before)
    assert any(not r["passed"] for r in final.challenge_results)


def test_genuine_high_risk_request_needs_one_liveness_check(runs):
    _, hist = runs["genuine_new_number"]
    before = [h for h in hist if not h.challenge_results]
    after = [h for h in hist if h.challenge_results]
    assert all(h.action == "STEP_UP" for h in before), "SIM change waits for a liveness check"
    assert all(h.risk < 40 for h in before[2:]), "while the voice itself already looks genuine"
    assert after and all(h.action == "PROCEED" for h in after)
    assert all(r["passed"] for r in after[-1].challenge_results)


def test_bot_talks_over_the_agent(runs):
    final = runs["ai_negotiator"][1][-1]
    assert final.signals["behavior"].details["barge_in_overrun_s"] > 0.8
    assert final.signals["voiceprint"].details["matched"] is False


def test_genuine_caller_yields_when_interrupted(runs):
    final = runs["genuine_customer"][1][-1]
    assert final.signals["behavior"].details["barge_in_overrun_s"] < 0.5


def test_disclosed_assistant_is_not_treated_as_fraud(runs):
    _, hist = runs["ai_assistant"]
    spoken = [h for h in hist if h.turn_idx >= 1]
    assert all(h.action == "AUTHORIZED_AGENT" for h in spoken)
    assert hist[1 + 1].verdict == "Disclosed AI assistant"           # before the customer approved
    assert hist[-1].verdict == "AI assistant - customer consent confirmed"


def test_handoff_card_skips_reverification_only_for_verified_callers(runs):
    for cid, (sess, hist) in runs.items():
        card = handoff.handoff_card(sess.call, sess.account, hist[-1], hist)
        assert ("NO re-verification needed" in card) == (cid.startswith("genuine")), cid


def test_every_caller_turn_has_model_scores():
    for cid in list_calls():
        for t in load_call(cid)["turns"]:
            if t["speaker"] == "caller":
                assert set(t["scores"]) == {"synthetic_voice", "voice_similarity"}, (cid, t["idx"])


def test_human_and_synthetic_callers_are_separated_by_the_models():
    for cid in list_calls():
        call = load_call(cid)
        p = [t["scores"]["synthetic_voice"] for t in call["turns"] if t["speaker"] == "caller"][-1]
        assert (p < 0.3) if call["ground_truth"] == "human" else (p > 0.7), cid
