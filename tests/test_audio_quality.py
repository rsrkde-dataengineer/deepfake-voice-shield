"""Quality gate for every committed recording, so broken or 'buzzing' audio can never ship again.

The old synthetic demo audio fails this on purpose: under 1 % of its energy was above 2 kHz
(no consonants), where real speech has several percent.
"""
import pytest

from vshield.audio import PLAYBACK_SR, SR, load_wav, quality
from vshield.session import DATA, list_calls, load_accounts, load_call


def _files():
    out = []
    for cid in list_calls():
        call = load_call(cid)
        out += [(call["audio"]["call"], PLAYBACK_SR), (call["audio"]["caller"], SR)]
    out += [(a["voiceprint"], SR) for a in load_accounts().values()]
    return out


@pytest.mark.parametrize("rel,sr", _files())
def test_recording_is_clean_speech(rel, sr):
    y, rate = load_wav(str(DATA / rel))
    q = quality(y, rate)
    assert rate == sr
    assert q["clipped_share"] < 0.001, "clipping"
    assert -30 <= q["speech_level_db"] <= -12, "too quiet or too loud"
    assert q["speech_share"] > 0.5, "mostly silence"
    assert q["energy_above_2khz"] > 0.02, "muffled: no consonant energy (the old buzzing audio)"


@pytest.mark.parametrize("cid", list_calls())
def test_timeline_matches_the_audio(cid):
    call = load_call(cid)
    caller, _ = load_wav(str(DATA / call["audio"]["caller"]))
    full, _ = load_wav(str(DATA / call["audio"]["call"]))
    spans = [t["audio_span"] for t in call["turns"] if t["speaker"] == "caller"]
    assert abs(spans[-1][1] - len(caller) / SR) < 0.05
    assert all(a[1] == pytest.approx(b[0], abs=1e-3) for a, b in zip(spans, spans[1:]))
    assert len(full) / PLAYBACK_SR >= max(t["end"] for t in call["turns"])
