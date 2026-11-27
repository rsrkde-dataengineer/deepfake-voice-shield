"""Clone Pavithra with Chatterbox (the project's open-source cloning tool) from her short voice note.
Consent: Pavithra gave consent (2026-11-27). Output stays in this folder, outside the public repo.
Run with the project's .venv-gen:  python make_clone.py "text" out_name [seed ...]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
SAMPLES = Path("/Users/shivashankar/Downloads/Voice samples")
REF_SRC = SAMPLES / "Original voice - pavi.ogg"          # the short voice note an attacker could get hold of
REF_WAV = HERE / "work" / "ref_pavi_voice_note.wav"


def main():
    text, name, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]] or [1]
    if not REF_WAV.exists():
        y, sr = sf.read(REF_SRC, dtype="float32", always_2d=True)
        sf.write(REF_WAV, y.mean(axis=1), sr)
    from chatterbox.tts import ChatterboxTTS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=dev)
    for seed in seeds:
        torch.manual_seed(seed)
        wav = model.generate(text, audio_prompt_path=str(REF_WAV)).squeeze(0).cpu().numpy()
        out = HERE / "work" / f"{name}_seed{seed}.wav"
        sf.write(out, wav.astype(np.float32), model.sr)
        print(f"{out.name}: {len(wav) / model.sr:.1f} s", flush=True)


if __name__ == "__main__":
    main()
