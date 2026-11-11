"""Live models reproduce the stored scores. Runs only where requirements-ml.txt is installed (.venv-ml)."""
import pytest

from vshield import detectors
from vshield.session import CallSession, load_accounts, load_call

pytestmark = pytest.mark.skipif(not detectors.available(), reason="ML packages not installed (requirements-ml.txt)")


@pytest.mark.parametrize("cid", ["genuine_customer", "voice_clone"])
def test_live_scores_match_the_stored_ones(cid):
    call = load_call(cid)
    sess = CallSession(call, load_accounts(), live=True)
    while not sess.done:
        sess.add_turn()
        t = sess.turns[-1]
        if t.speaker == "caller":
            p, sim, _ = sess.model_scores()
            if p is not None:
                assert p == pytest.approx(t.scores["synthetic_voice"], abs=0.02)
                assert sim == pytest.approx(t.scores["voice_similarity"], abs=0.02)
    assert sess.history[-1].action == call["expected_action"]
