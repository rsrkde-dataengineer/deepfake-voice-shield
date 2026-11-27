"""'Real or clone?' game for the SVP: pick the rounds, loudness-match the clips, score each with Shield's two real
models against Pavithra's voiceprint, and write game.json + the clips (.m4a) for the site.

    cd <repo> && .venv-ml/bin/python "/Users/shivashankar/Downloads/Shield own voices/score_game.py"
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path("/Users/shivashankar/Downloads/deepfake-voice-shield")
sys.path.insert(0, str(REPO))
from vshield import detectors  # noqa: E402
from vshield.audio import SR, resample  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from own_calls import NEW, ROLE_PLAY, SAMPLES, VOICE_NOTE, read_any, tidy  # noqa: E402

WORK, SITE = HERE / "work", HERE / "site" / "Deepfake Voice Shield SVP"
MATCH_AT, MISMATCH_AT, SYNTH_AT = 0.70, 0.55, 0.5
SUN = "The sun began to rise over the quiet valley, casting long gold shadows across the sleeping village below."


def rounds(final: bool) -> list[dict]:
    final = False   # the game always uses Pavithra's pairs; the full calls use Rajkumar's session
    app_clone = {"src": SAMPLES / "pavis voice clone - more accurate.mp3", "truth": "clone",
                 "what": "An AI clone of Pavithra, made with a consumer voice-cloning app"}
    real_sun = {"src": VOICE_NOTE, "truth": "real", "what": "Pavithra, a real human (a voice note)"}
    open_clone_sun = {"src": WORK / "sun_clone_seed7.wav", "truth": "clone",
                      "what": "An AI clone of Pavithra, made with an open-source tool from her 10-second voice note"}
    if final:
        return [
            {"id": "r1", "title": "Round 1", "sentence": "Oh hi, yeah. My name is Jennifer Miller…",
             "clips": [{"key": "A", "src": WORK / "clone_line1.wav", "truth": "clone",
                        "what": "An AI clone of Pavithra, made with an open-source tool from her 10-second voice note"},
                       {"key": "B", "src": NEW["line 1"], "truth": "real", "what": "Pavithra, a real human"}]},
            {"id": "r2", "title": "Round 2", "sentence": SUN,
             "clips": [{"key": "A", **real_sun}, {"key": "B", **app_clone}]},
        ]
    return [
        {"id": "r1", "title": "Round 1", "sentence": SUN, "clips": [{"key": "A", **app_clone}, {"key": "B", **real_sun}]},
        {"id": "r2", "title": "Round 2", "sentence": SUN, "clips": [{"key": "A", **real_sun}, {"key": "B", **open_clone_sun}]},
    ]


def conclusion(p: float, sim: float) -> tuple[str, str]:
    """What Shield concludes from the voice alone: (action, verdict)."""
    if p >= SYNTH_AT:
        return "ESCALATE", ("AI voice clone that matches her voiceprint" if sim >= MATCH_AT else "AI-generated voice")
    if sim >= MATCH_AT:
        return "PROCEED", "Live human voice that matches Pavithra's voiceprint"
    if sim <= MISMATCH_AT:
        return "STEP_UP", "A human-sounding voice, but not Pavithra's voiceprint"
    return "STEP_UP", "Human voice, voiceprint not confirmed: one quick check"


def main() -> None:
    final = all(p.exists() for p in NEW.values())     # Rajkumar's session in: the calls are final
    enroll_src = ROLE_PLAY                              # the game scores Pavithra against her own 31 s recording
    detectors.load()
    y, sr = read_any(enroll_src)
    enrolled = resample(y, sr, SR)
    out = {"provisional": not final, "person": "Pavithra", "enrolled": f"Pavithra's voiceprint on file ({len(enrolled) / SR:.0f} s recording)",
           "rounds": []}
    SITE.mkdir(parents=True, exist_ok=True)
    for rd in rounds(final):
        clips = []
        for c in rd["clips"]:
            y, sr = read_any(Path(c["src"]))
            y = tidy(resample(y, sr, 24_000), 24_000)                 # same loudness for every clip
            wav = WORK / f"game_{rd['id']}_{c['key']}.wav"
            sf.write(wav, y, 24_000)
            name = f"game_{rd['id']}_{c['key'].lower()}.m4a"
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "64000", str(wav), str(SITE / name)], check=True)
            y16 = resample(y, 24_000, SR)
            p, sim = detectors.synthetic_probability(y16), detectors.voice_similarity(y16, enrolled)
            action, verdict = conclusion(p, sim)
            clips.append({"key": c["key"], "file": name, "truth": c["truth"], "what": c["what"], "seconds": round(len(y) / 24_000, 1),
                          "p": round(p, 4), "pText": f"{p:.0%}", "sim": round(sim, 3), "simText": f"{sim:.2f}",
                          "match": sim >= MATCH_AT, "ai": p >= SYNTH_AT, "action": action, "verdict": verdict})
            print(f"{rd['id']} {c['key']} {c['truth']:5s} synthetic {p:6.1%} voiceprint {sim:5.2f} -> {action:8s} {verdict}")
        out["rounds"].append({k: rd[k] for k in ("id", "title", "sentence")} | {"clips": clips})
    (HERE / "game.json").write_text(json.dumps(out, indent=2) + "\n")
    print("game.json written;", "FINAL" if final else "PROVISIONAL", "enrollment:", enroll_src.name)


if __name__ == "__main__":
    main()
