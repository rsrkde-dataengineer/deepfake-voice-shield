"""Clone a voice from a short recording, for live demos of the detector (run in .venv-gen).

    .venv-gen/bin/python scripts/clone_voice.py my_voice.wav            # -> my_voice_clone.wav
    .venv-gen/bin/python scripts/clone_voice.py my_voice.wav --text "Hi, I need a new SIM card today."

Only clone your own voice, or a teammate's with their written consent. The output carries
Resemble AI's Perth watermark (Chatterbox, MIT licence).

Demo: on the dashboard's "Try your voice" page, enroll yourself with the microphone, then upload the
clone: the voiceprint matches you, and Shield flags it as a voice clone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vshield.audio import SR, resample, save_wav  # noqa: E402

DEFAULT_TEXT = ("Hi, my phone was stolen this morning and I need to move my number to a new SIM card right away. "
                "I can't access my email, so please send the code somewhere else.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference", help="10-30 s WAV/FLAC of the voice to clone")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", help="output WAV (default: <reference>_clone.wav)")
    args = ap.parse_args()

    import torch
    from chatterbox.tts import ChatterboxTTS

    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=dev)
    torch.manual_seed(0)
    y = model.generate(args.text, audio_prompt_path=args.reference).squeeze(0).cpu().numpy()
    y = resample(y, model.sr, SR)
    out = args.out or str(Path(args.reference).with_name(Path(args.reference).stem + "_clone.wav"))
    save_wav(out, 0.9 * y / (np.abs(y).max() + 1e-9), SR)
    print(f"Clone written to {out} ({len(y) / SR:.1f} s)")


if __name__ == "__main__":
    main()
