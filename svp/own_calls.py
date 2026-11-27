"""Build two own-voice demo calls (a team member, recorded and cloned with consent) in a separate data folder, so the
project's data/ and its 7 verified calls stay untouched.

    data_own/ = copy of the project's data (accounts, 7 calls, audio, enrollment)
              + account ACC-5001 (a fictional customer the team member plays), voiceprint = the enrollment recording
              + call team_real  : the real voice asks to move the number to a new phone
              + call team_clone : the same words in a Chatterbox clone of that voice, made from a short voice note

Uses Rajkumar's session ('raj enroll / line 1..4 / voice note.m4a' in Voice samples) when all six exist; otherwise
stand-ins from Pavithra's earlier clips, marked PROVISIONAL in the call files.

    cd <repo> && .venv-gen/bin/python "/Users/shivashankar/Downloads/Shield own voices/own_calls.py"
    VSHIELD_DATA="<this folder>/data_own" .venv-ml/bin/python scripts/score_calls.py team_real team_clone
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path("/Users/shivashankar/Downloads/deepfake-voice-shield")
sys.path.insert(0, str(REPO))
from vshield.audio import PLAYBACK_SR, SR, resample, save_wav  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLES = Path("/Users/shivashankar/Downloads/Voice samples")
OUT, WORK = HERE / "data_own", HERE / "work"
NEW = {k: SAMPLES / f"raj {k}.m4a" for k in ("enroll", "line 1", "line 2", "line 3", "line 4", "voice note")}
VOICE_NOTE = SAMPLES / "Original voice - pavi.ogg"         # stand-in only: Pavithra's 10 s voice note
ROLE_PLAY = SAMPLES / "genuine voice - pavi.mp3"           # stand-in for the call lines
ACCOUNT, PHRASE = "ACC-5001", "silver maple eighteen"
AGENT_NAME, AGENT_VOICE = "Jessica", "af_jessica"
LINES = ["Oh hi, yeah. My name is David Miller. I recently upgraded to a new phone, and, um, I want to move my number "
         "to the new device.",
         "I tried activating the eSIM myself, but, uh, it keeps failing.",
         "Silver maple eighteen.",
         "Oh great, thank you so much."]
# Pavithra may have recorded the first wording (no little words); set FIRST_WORDING=1 in the environment if so.
if __import__("os").environ.get("FIRST_WORDING"):
    LINES = ["Hello, my name is David Miller. I recently upgraded to a new phone and I want to move my number to the "
             "new device.", "I tried activating the eSIM myself, but it keeps failing.", "Silver maple eighteen.",
             "Great, thank you so much."]


# ---------------------------------------------------------------------------------------------- audio helpers
def read_any(path: Path):
    """(mono float32, sr). m4a via macOS afconvert; tolerates MP3s with a damaged tail."""
    if path.suffix.lower() in (".m4a", ".aac", ".caf", ".mp4"):
        wav = WORK / (path.stem + ".wav")
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(wav)], check=True)
        path = wav
    blocks = []
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        while True:
            try:
                b = f.read(sr, dtype="float32", always_2d=True)
            except sf.LibsndfileError:
                break
            if not len(b):
                break
            blocks.append(b)
    return np.concatenate(blocks).mean(axis=1), sr


def tidy(y: np.ndarray, sr: int, level_db: float = -20.0) -> np.ndarray:
    """Trim leading/trailing silence and bring speech to a common loudness (same as build_demo_calls.tidy)."""
    frame = int(0.02 * sr)
    n = len(y) // frame
    rms = np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    keep = np.where(db > max(db.max() - 40, -50))[0]
    pad = int(0.08 * sr)
    y = y[max(keep[0] * frame - pad, 0): min((keep[-1] + 1) * frame + pad, len(y))]
    active = rms[keep[0]: keep[-1] + 1]
    y = y * 10 ** ((level_db - 20 * np.log10(np.sqrt((active ** 2).mean()))) / 20)
    return (y / max(1.0, np.abs(y).max() / 0.95)).astype(np.float32)


def last_pause_end(y: np.ndarray, sr: int, min_pause: float = 0.3) -> int:
    """Sample index where the last pause of at least min_pause starts (to end a clip on a clean boundary)."""
    frame = int(0.02 * sr)
    n = len(y) // frame
    db = 20 * np.log10(np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12))
    quiet = db < db.max() - 35
    run, best = 0, n
    for i in range(n):
        run = run + 1 if quiet[i] else 0
        if run * 0.02 >= min_pause:
            best = i - run + 1
    return best * frame


# ---------------------------------------------------------------------------------------------- voices
def main() -> None:
    import torch
    from chatterbox.tts import ChatterboxTTS
    from kokoro import KPipeline

    WORK.mkdir(parents=True, exist_ok=True)
    final = all(p.exists() for p in NEW.values())
    person, customer = ("Rajkumar", "David Miller") if final else ("Pavithra", "Jennifer Miller")
    print("recordings:", "Rajkumar's session (final)" if final else "PROVISIONAL stand-ins (Pavithra's earlier clips)")

    # 1. data_own = a copy of the project's data, then our additions
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(REPO / "data", OUT, ignore=shutil.ignore_patterns("external", "samples", ".DS_Store"))

    # 2. Pavithra's real voice: enrollment and call lines
    if final:
        enroll_y, enroll_sr = read_any(NEW["enroll"])
        real_lines = [tidy(resample(*read_any(NEW[f"line {i}"]), PLAYBACK_SR), PLAYBACK_SR) for i in range(1, 5)]
        line_text = LINES
        source = "Rajkumar (team member): enrollment and the four call lines, one session on one phone"
    else:
        enroll_y, enroll_sr = read_any(VOICE_NOTE)
        y, sr = read_any(ROLE_PLAY)
        y = y[: last_pause_end(y, sr)]                     # end on a clean pause before the damaged tail
        real_lines = [tidy(resample(y, sr, PLAYBACK_SR), PLAYBACK_SR)]
        line_text = ["Hello, my name is Jennifer Miller. I've recently upgraded to your new mobile phone and I wanted to "
                     "transfer my existing number to the new device. I actually tried activating the eSIM, but the "
                     "process is not completing successfully."]
        source = "PROVISIONAL: enrollment = her 10 s voice note (also the clone's reference); call = her role-play clip"
    save_wav(str(OUT / "enrollment" / f"{ACCOUNT}.wav"), tidy(resample(enroll_y, enroll_sr, SR), SR), SR)

    # 3. the clone: Chatterbox from a short voice note (a separate take), the same words as the real lines
    ref = WORK / ("ref_raj_voice_note.wav" if final else "ref_pavi_voice_note.wav")
    y, sr = read_any(NEW["voice note"] if final else VOICE_NOTE)
    sf.write(ref, y, sr)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cloner = ChatterboxTTS.from_pretrained(device=dev)
    clone_text = line_text if final else line_text + LINES[2:]
    clone_lines = []
    for i, text in enumerate(clone_text):
        torch.manual_seed(5000 + i)
        y = cloner.generate(text, audio_prompt_path=str(ref)).squeeze(0).cpu().numpy()
        clone_lines.append(tidy(resample(y, cloner.sr, PLAYBACK_SR), PLAYBACK_SR))
        sf.write(WORK / f"clone_line{i + 1}.wav", clone_lines[-1], PLAYBACK_SR)     # for the game
        print(f"  clone line {i + 1}: {len(clone_lines[-1]) / PLAYBACK_SR:.1f} s", flush=True)

    kokoro = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def agent(text: str) -> np.ndarray:
        chunks = [np.asarray(a) for _, _, a in kokoro(text, voice=AGENT_VOICE)]
        return tidy(np.concatenate(chunks).astype(np.float32), PLAYBACK_SR)

    greet = f"Thank you for calling, this is {AGENT_NAME}. How can I help you today?"
    ack = "Congratulations on the new phone. I can help you move your number."
    ask = ("No problem, I can do that for you. Moving a number is a high-risk change, so first, could you please "
           f"repeat this phrase for me: {PHRASE}.")
    if final:
        real_steps = [("agent", greet), ("caller", 0, 0.95), ("agent", ack), ("caller", 1, 1.30),
                      ("agent", ask, "challenge"), ("caller", 2, 0.85),
                      ("agent", "Thank you, that's confirmed. Your number is moving to the new device now; you'll get "
                                "a text in a few minutes."), ("caller", 3, 1.05)]
        clone_steps = [("agent", greet), ("caller", 0, 1.14), ("agent", ack), ("caller", 1, 1.18),
                       ("agent", ask, "challenge"), ("caller", 2, 2.55),
                       ("agent", "Thank you. One moment while I review the request."), ("caller", 3, 1.16)]
    else:
        real_steps = [("agent", greet), ("caller", 0, 0.95)]      # no stand-in recording answers the phrase: don't ask it
        clone_steps = [("agent", greet), ("caller", 0, 1.14), ("agent", ask, "challenge"), ("caller", 1, 2.55),
                       ("agent", "Thank you. One moment while I review the request."), ("caller", 2, 1.16)]

    def build(cid: str, steps, caller_audio, texts, telecom, meta):
        mix, caller_clips, turns, t, pos = np.zeros(int(200 * PLAYBACK_SR), np.float32), [], [], 0.3, 0.0
        for i, step in enumerate(steps):
            if step[0] == "agent":
                y = agent(step[1])
                start = t if i == 0 else t + 0.45
                turn = {"idx": i, "speaker": "agent", "text": step[1], "start": round(start, 2),
                        "end": round(start + len(y) / PLAYBACK_SR, 2)}
                if len(step) > 2:
                    turn["challenge"] = {"type": "phrase_repeat", "phrase": PHRASE}
            else:
                y, start = caller_audio[step[1]], t + step[2]
                clip = resample(y, PLAYBACK_SR, SR)
                turn = {"idx": i, "speaker": "caller", "text": texts[step[1]], "start": round(start, 2),
                        "end": round(start + len(y) / PLAYBACK_SR, 2), "source": meta["source"],
                        "audio_span": [round(pos, 3), round(pos + len(clip) / SR, 3)]}
                caller_clips.append(clip)
                pos += len(clip) / SR
            a = int(start * PLAYBACK_SR)
            mix[a: a + len(y)] += y
            turns.append(turn)
            t = start + len(y) / PLAYBACK_SR
        mix = mix[: int((t + 0.6) * PLAYBACK_SR)]
        save_wav(str(OUT / "audio" / f"{cid}.wav"), mix / max(1.0, np.abs(mix).max() / 0.95), PLAYBACK_SR)
        save_wav(str(OUT / "audio" / f"{cid}_caller.wav"), np.concatenate(caller_clips), SR)
        call = {"call_id": cid, "title": meta["title"], "summary": meta["summary"], "account_id": ACCOUNT,
                "ground_truth": meta["truth"], "expected_action": meta["expected"], "agent_name": AGENT_NAME,
                "challenge_phrase": PHRASE, "voices": {"caller": meta["voice"], "agent": f"Kokoro TTS voice {AGENT_VOICE}"},
                "audio": {"call": f"audio/{cid}.wav", "caller": f"audio/{cid}_caller.wav"}, "telecom": telecom,
                "turns": turns, "provisional": not final, "person": meta["person"]}
        (OUT / "calls" / f"{cid}.json").write_text(json.dumps(call, indent=2) + "\n")
        print(f"{cid}: {len(turns)} turns, {t:.1f} s", flush=True)

    base = {"cti_destination": "CARE_DEVICE_SIM", "ivr_intent": "sim_change"}
    build("team_real", real_steps, real_lines, line_text,
          {"ani": "+1-312-555-0147", "stir_shaken": "A", "line_type": "mobile", "carrier": "Home network",
           "calls_from_ani_24h": 1, "accounts_called_from_ani_7d": 1, **base, "ivr_dtmf_timing_ms": [1210, 840, 1490, 1020]},
          {"title": f"Real voice: {person} moves a number to a new phone",
           "summary": f"Team member {person}, a real person, plays customer {customer} and asks to move the number to a "
                      "new phone. Shield checks the voice against the voiceprint on file and " + ("asks for a liveness phrase." if final
                      else "suggests the next check."),
           "truth": "human", "expected": "PROCEED", "source": f"{person} (real voice)", "person": person,
           "voice": f"Real human voice: {person}, team member, recorded with consent"})
    build("team_clone", clone_steps, clone_lines, clone_text,
          {"ani": "+1-646-555-0182", "stir_shaken": "C", "line_type": "voip", "carrier": "Unknown VoIP gateway",
           "calls_from_ani_24h": 3, "accounts_called_from_ani_7d": 2, **base, "ivr_dtmf_timing_ms": [910, 1170, 1010, 1230]},
          {"title": f"AI clone of {person}: the same request, the same words",
           "summary": f"A fraudster clones {person}'s voice from a short voice note and makes the same request in the same "
                      "words. Shield hears the synthetic speech and the clone fails the liveness phrase.",
           "truth": "voice_clone", "expected": "ESCALATE", "source": "Chatterbox clone", "person": person,
           "voice": f"Chatterbox voice clone of {person}, made from a short voice note (with consent)"})

    accounts = json.loads((OUT / "accounts.json").read_text())
    accounts[ACCOUNT] = {
        "name": customer, "tenure_years": 6, "plan": "Unlimited Plus (2 lines)", "anis_on_file": ["+1-312-555-0147"],
        "risk_flags": [], "security_events_7d": [], "billing_history": {"credits_90d": 0, "retention_offers_12m": 0,
                                                                         "last_bill": 112.3},
        "active_tickets": [], "records": [{"id": "device", "question": "Which phone did you get from us last week?",
                                           "answer": "Galaxy S25", "accept": ["galaxy", "s25"]}],
        "voiceprint": f"enrollment/{ACCOUNT}.wav", "voiceprint_source": source}
    (OUT / "accounts.json").write_text(json.dumps(accounts, indent=2) + "\n")
    print("data_own ready:", OUT)


if __name__ == "__main__":
    main()
