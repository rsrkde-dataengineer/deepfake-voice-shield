"""Unit tests for the individual signals, the decision policy and the impact model."""
import numpy as np

from vshield import challenge, fusion, impact
from vshield.audio import telephone
from vshield.models import SignalResult, Turn
from vshield.signals import behavior, context, synthetic_voice, voiceprint

ACCOUNT = {"name": "Test Person", "anis_on_file": ["+1-201-555-0100"]}


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


def test_telephone_channel_is_narrowband():
    from scipy.signal import welch
    y = np.random.default_rng(0).uniform(-0.5, 0.5, 16_000).astype(np.float32)
    z = telephone(y)
    f, p = welch(z, 16_000)
    assert abs(len(z) - len(y)) <= 2
    assert p[f > 4200].sum() < 0.01 * p.sum()
