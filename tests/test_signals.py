"""Unit tests for the individual signals, the decision policy and the impact model."""
import numpy as np

from vshield import challenge, fusion, identity, impact
from vshield.audio import telephone
from vshield.models import SignalResult, Turn
from vshield.signals import behavior, context, synthetic_voice, voiceprint

ACCOUNT = {"name": "Test Person", "anis_on_file": ["+1-201-555-0100"],
           "records": [{"id": "device", "question": "Which phone did you get from us in April?", "answer": "Galaxy S25",
                        "accept": ["galaxy", "samsung"]},
                       {"id": "travel", "question": "Where did you last use TravelPass?", "answer": "Cancun",
                        "accept": ["cancun"]}]}


def test_latency_regularity():
    turns = [Turn(0, "agent", "hi", 0, 2), Turn(1, "caller", "a b c", 2.9, 4), Turn(2, "agent", "x", 4.5, 6),
             Turn(3, "caller", "d e f", 6.9, 8), Turn(4, "agent", "y", 8.5, 10), Turn(5, "caller", "g", 10.9, 12)]
    assert behavior.caller_latencies(turns) == [0.9, 0.9, 0.9]
    assert behavior.analyze(turns).score > 0.6


def test_challenge_latency_spike_fails_even_with_the_right_words():
    turns = [Turn(0, "agent", "repeat: red fox", 0, 2, challenge={"type": "phrase_repeat", "phrase": "red fox"}),
             Turn(1, "caller", "Red fox.", 4.8, 5.5)]
    r = challenge.evaluate(turns, [0.9, 0.9])[0]
    assert r["content_ok"] and not r["passed"]


def test_challenge_prompt_answer_passes():
    turns = [Turn(0, "agent", "repeat: red fox", 0, 2, challenge={"type": "phrase_repeat", "phrase": "red fox"}),
             Turn(1, "caller", "Red fox.", 3.0, 3.6)]
    assert challenge.evaluate(turns, [0.9, 1.1])[0]["passed"]


def test_suggested_phrase_comes_from_the_call():
    assert challenge.suggest({"call_id": "x", "challenge_phrase": "it is most delightful"}, 0)["phrase"] == "it is most delightful"
    assert challenge.suggest({"call_id": "x"}, 0)["phrase"] in challenge.PHRASES


def test_ai_disclosure_routes_to_the_authorised_agent_flow():
    turns = [Turn(0, "agent", "hello", 0, 1), Turn(1, "caller", "Hi, I am an AI assistant calling on behalf of Sam", 1.5, 4)]
    sig = {"behavior": behavior.analyze(turns), "telecom": SignalResult("telecom", "t", 0.3, 0.6)}
    assert fusion.fuse(sig).action == "AUTHORIZED_AGENT"
    assert fusion.fuse(sig, consent_approved=True).verdict == "AI assistant - customer consent confirmed"


def test_bot_farm_telecom():
    meta = {"ani": "+1", "stir_shaken": "B", "line_type": "voip", "calls_from_ani_24h": 20,
            "accounts_called_from_ani_7d": 9, "ivr_dtmf_timing_ms": [300, 300, 300]}
    assert context.telecom(meta, {"anis_on_file": []}).score > 0.8


def test_model_scores_become_explainable_signals():
    human = synthetic_voice.analyze(0.02, 6.0)
    fake = synthetic_voice.analyze(0.98, 6.0)
    assert human.score < 0.1 < 0.9 < fake.score and "live human" in human.reasons[0] and "synthetic" in fake.reasons[0]
    assert not synthetic_voice.analyze(0.9, 1.0).available          # too little speech yet
    match = voiceprint.analyze(0.85, 6.0, ACCOUNT)
    other = voiceprint.analyze(0.05, 6.0, ACCOUNT)
    assert match.details["matched"] and match.score == 0.1
    assert not other.details["matched"] and other.score == 0.9
    assert not voiceprint.analyze(0.85, 1.5, ACCOUNT).available


def _voice_signals(p_synth, similarity):
    return {"synthetic_voice": synthetic_voice.analyze(p_synth, 8.0),
            "voiceprint": voiceprint.analyze(similarity, 8.0, ACCOUNT),
            "telecom": SignalResult("telecom", "Telecom", 0.1, 0.6, ["Known number"]),
            "account": SignalResult("account", "Account", 0.1, 0.6, ["Clean history"])}


def test_high_risk_request_needs_liveness_even_for_a_genuine_voice():
    sig = _voice_signals(0.02, 0.88)
    assert fusion.fuse(sig).action == "PROCEED"
    assert fusion.fuse(sig, high_risk_request=True).action == "STEP_UP"
    assert fusion.fuse(sig, high_risk_request=True, liveness_passed=True).action == "PROCEED"


def test_matching_voice_with_synthetic_speech_is_a_clone():
    d = fusion.fuse(_voice_signals(0.98, 0.85))
    assert d.verdict == "Likely voice clone of the account holder" and d.risk >= 72


def test_impact_model_reproduces_the_brief():
    hours, dollars = impact.transfer_savings(**impact.BRIEF)
    assert 75_000 < hours < 77_000 and 2.2e6 < dollars < 2.35e6
    assert impact.fraud_avoided(100, 50, 0.9, 0.4) == 100 * 12 * 50 * 0.5
    assert 3.75e6 < hours * impact.FULLY_LOADED_HOURLY_COST < 3.85e6          # the pitch's "$3.8M fully loaded"
    assert round(impact.fraud_avoided(**impact.FRAUD_WHAT_IF)) == 12_600_000  # the pitch's illustrative "$12.6M"


def test_build_cost_and_payback_match_the_pitch():
    assert impact.build_cost() == (460, 120, 55_000)                          # a Large effort: 580 h, $55k
    _, dollars = impact.transfer_savings(**impact.BRIEF)
    assert round(impact.payback_days(dollars)) == 9                           # "about 9 days"
    assert round(impact.payback_days(dollars * impact.FULLY_LOADED_HOURLY_COST / impact.BRIEF["hourly_cost"])) == 5
    assert impact.payback_days(impact.RUN_COST_PER_YEAR) == float("inf")


def test_telephone_channel_is_narrowband():
    from scipy.signal import welch
    y = np.random.default_rng(0).uniform(-0.5, 0.5, 16_000).astype(np.float32)
    z = telephone(y)
    f, p = welch(z, 16_000)
    assert abs(len(z) - len(y)) <= 2
    assert p[f > 4200].sum() < 0.01 * p.sum()


def _asked(answer: str) -> list[Turn]:
    return [Turn(0, "agent", "Which phone did you get from us in April?", 0, 2, question={"record": "device"}),
            Turn(1, "caller", answer, 2.8, 4.0)]


def test_account_history_answers_are_matched_against_the_record():
    ok, = identity.evaluate(_asked("Oh, the Samsung Galaxy, I think."), ACCOUNT)
    dodge, = identity.evaluate(_asked("Oh, I would not venture to say, sir."), ACCOUNT)
    wrong, = identity.evaluate(_asked("An iPhone."), ACCOUNT)
    assert ok["passed"] and not dodge["passed"] and not wrong["passed"]
    assert "could not answer" in dodge["verdict"] and "does not match" in wrong["verdict"]
    assert identity.as_signal([ok]).score < 0.1 < 0.9 <= identity.as_signal([wrong]).score


def test_account_holder_saying_not_me_on_their_phone_is_the_strongest_identity_evidence():
    turns = [Turn(0, "agent", "The account holder has declined.", 0, 2, event="owner_denied")]
    sig = identity.as_signal(identity.evaluate(turns, ACCOUNT))
    assert sig.details["owner_denied"] and sig.score > 0.95 and "Not me" in sig.reasons[0]


def test_identity_question_suggestions_come_from_the_account_history():
    first = identity.suggest(ACCOUNT, set())
    assert first["type"] == "record_question" and "April" in first["agent_script"] and "Galaxy S25" in first["why"]
    assert identity.suggest(ACCOUNT, {"device"})["record"] == "travel"
    assert identity.suggest(ACCOUNT, {"device", "travel"})["type"] == "phone_confirmation"


def test_a_real_human_voice_that_is_not_the_account_holder_is_never_genuine():
    d = fusion.fuse(_voice_signals(0.02, 0.05))          # live human, voiceprint mismatch
    assert d.action == "STEP_UP" and d.verdict == "Not the account holder's voice" and d.pattern == "other_voice"
    assert fusion.fuse(_voice_signals(0.02, 0.62)).verdict == "Identity not confirmed - weak voiceprint match"
    assert fusion.fuse(_voice_signals(0.02, 0.88)).verdict == "Likely genuine customer"


def test_failed_identity_check_on_someone_elses_voice_escalates():
    failed = identity.as_signal(identity.evaluate(_asked("I don't know."), ACCOUNT))
    sig = _voice_signals(0.02, 0.05) | {"identity": failed}
    d = fusion.fuse(sig)
    assert d.action == "ESCALATE" and d.risk >= 88 and d.verdict == "Likely impersonator - not the account holder"
    sig["account"] = SignalResult("account", "Account", 0.1, 0.6, ["Clean history"], {"intent": "retention"})
    assert fusion.fuse(sig).verdict == "Likely bill negotiator posing as the customer"
    matched = fusion.fuse(_voice_signals(0.02, 0.88) | {"identity": failed})
    assert matched.action == "PROCEED", "the real customer forgetting a detail is not fraud"


def test_human_saying_on_behalf_of_is_not_an_ai_disclosure():
    turns = [Turn(0, "agent", "hello", 0, 1), Turn(1, "caller", "I'm calling on behalf of my client", 1.5, 4)]
    assert not behavior.analyze(turns).details["ai_disclosed"]


def test_strongest_telecom_clue_comes_first():
    meta = {"ani": "+1", "stir_shaken": "A", "line_type": "voip", "calls_from_ani_24h": 41,
            "accounts_called_from_ani_7d": 37}
    assert context.telecom(meta, {"anis_on_file": []}).reasons[0].startswith("Same number touched 37")
