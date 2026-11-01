import numpy as np
import pytest

from vshield import challenge, fusion
from vshield.models import SignalResult, Turn
from vshield.session import CallSession, load_accounts, load_call
from vshield.signals import acoustic, behavior, context
from vshield.synth import HUMAN, TTS, Speaker, say


@pytest.fixture(scope="module")
def results():
    acc = load_accounts()
    return {cid: CallSession(load_call(cid), acc).run() for cid in ("genuine_customer", "voice_clone", "ai_negotiator")}


def test_genuine_customer_passes(results):
    final = results["genuine_customer"][-1]
    assert final.risk < 20
    assert final.action == "PROCEED"
    assert final.signals["voiceprint"].details["matched"]


def test_voice_clone_detected_despite_voiceprint_match(results):
    final = results["voice_clone"][-1]
    assert final.signals["voiceprint"].details["matched"], "clone should fool the voiceprint alone"
    assert final.signals["acoustic"].score > 0.75
    assert final.verdict == "Likely voice clone of the account holder"
    assert final.action == "ESCALATE"


def test_clone_is_step_up_before_challenge(results):
    hist = results["voice_clone"]
    pre = [h for h in hist if not h.challenge_results and h.turn_idx >= 1]
    assert any(h.action == "STEP_UP" and h.suggested_challenge for h in pre)


def test_ai_bot_detected(results):
    final = results["ai_negotiator"][-1]
    assert final.verdict == "Likely AI agent / bot"
    assert final.action == "ESCALATE"
    assert any(not r["passed"] for r in final.challenge_results)
    assert final.signals["behavior"].details["barge_in_overrun_s"] > 0.8


def test_never_claims_certainty(results):
    for hist in results.values():
        assert all(1 <= h.risk <= 99 for h in hist)


def test_acoustic_separates_human_and_tts():
    spk = Speaker("t", 150, 1.05, 0.95)
    txt = "I am calling about my monthly bill and a charge I do not recognise on my statement"
    h = acoustic.analyze(say(txt, spk, HUMAN, 3)).score
    t = acoustic.analyze(say(txt, spk, TTS, 3)).score
    assert h < 0.3 < 0.7 < t


def test_latency_regularity():
    turns = [Turn(0, "agent", "hi", 0, 2), Turn(1, "caller", "a b c", 2.9, 4), Turn(2, "agent", "x", 4.5, 6),
             Turn(3, "caller", "d e f", 6.9, 8), Turn(4, "agent", "y", 8.5, 10), Turn(5, "caller", "g", 10.9, 12)]
    assert behavior._caller_latencies(turns) == [0.9, 0.9, 0.9]


def test_challenge_latency_spike_fails():
    turns = [Turn(0, "agent", "repeat: red fox", 0, 2, challenge={"type": "phrase_repeat", "phrase": "red fox"}),
             Turn(1, "caller", "Red fox.", 4.8, 5.5)]
    r = challenge.evaluate(turns, [0.9, 0.9])[0]
    assert r["content_ok"] and not r["passed"]


def test_ai_disclosure_routes_to_authorised_flow():
    turns = [Turn(0, "agent", "hello", 0, 1), Turn(1, "caller", "Hi, I am an AI assistant calling on behalf of Sam", 1.5, 4)]
    sig = {"behavior": behavior.analyze(turns),
           "telecom": SignalResult("telecom", "t", 0.3, 0.6)}
    _, verdict, action, _ = fusion.fuse(sig)
    assert action == "AUTHORIZED_AGENT"


def test_bot_farm_telecom():
    meta = {"ani": "+1", "stir_shaken": "B", "line_type": "voip", "calls_from_ani_24h": 20,
            "accounts_called_from_ani_7d": 9, "ivr_dtmf_timing_ms": [300, 300, 300]}
    assert context.telecom(meta, {"anis_on_file": []}).score > 0.8
