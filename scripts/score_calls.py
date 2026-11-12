"""Run the real models over every demo call and store their scores in the call files (run in .venv-ml).

    .venv-ml/bin/python scripts/score_calls.py                       # every call
    .venv-ml/bin/python scripts/score_calls.py human_negotiator      # or only the calls you name

For each caller turn, Spectra-AASIST and NVIDIA TitaNet score all caller audio up to the end of that turn
(exactly what the live session does), and the result is saved as turn["scores"]. The dashboard then shows
real model output on any laptop, without installing torch or downloading the models.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vshield import detectors  # noqa: E402
from vshield.audio import SR, load_wav  # noqa: E402
from vshield.session import DATA, list_calls, load_accounts  # noqa: E402


def main(only: list[str]) -> None:
    accounts = load_accounts()
    detectors.load()
    for cid in only or list_calls():
        path = DATA / "calls" / f"{cid}.json"
        call = json.loads(path.read_text())
        audio, _ = load_wav(str(DATA / call["audio"]["caller"]))
        enrolled, _ = load_wav(str(DATA / accounts[call["account_id"]]["voiceprint"]))
        t0 = time.time()
        for turn in call["turns"]:
            if turn["speaker"] != "caller":
                continue
            y = audio[: int(turn["audio_span"][1] * SR)]
            turn["scores"] = {"synthetic_voice": round(detectors.synthetic_probability(y), 4),
                              "voice_similarity": round(detectors.voice_similarity(y, enrolled), 4)}
        call["scored_with"] = {**detectors.MODEL_NAMES, "date": date.today().isoformat()}
        path.write_text(json.dumps(call, indent=2) + "\n")
        last = [t["scores"] for t in call["turns"] if t["speaker"] == "caller"][-1]
        print(f"  {cid:20s} synthetic {last['synthetic_voice']:.2f}  voiceprint {last['voice_similarity']:.2f}"
              f"  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
