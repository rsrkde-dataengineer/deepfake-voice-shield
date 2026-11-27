"""Score any clips against Pavithra's enrolled voiceprint with Shield's two real models (read-only use of the project).
Run with the project's .venv-ml:  python score_clips.py clip1 clip2 ...
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/shivashankar/Downloads/deepfake-voice-shield")
from vshield import detectors  # noqa: E402
from vshield.audio import SR, resample, telephone  # noqa: E402

ENROLL = Path("/Users/shivashankar/Downloads/Voice samples/genuine voice - pavi.mp3")


def load(path) -> np.ndarray:
    blocks = []
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        while True:
            try:
                b = f.read(sr, dtype="float32", always_2d=True)
            except sf.LibsndfileError:
                break                      # damaged tail: keep what decoded
            if not len(b):
                break
            blocks.append(b)
    return resample(np.concatenate(blocks).mean(axis=1), sr, SR)


def main():
    detectors.load()
    enrolled = load(ENROLL)
    enrolled_phone = telephone(enrolled)
    for p in sys.argv[1:]:
        y = load(p)
        s, v = detectors.synthetic_probability(y), detectors.voice_similarity(y, enrolled)
        yp = telephone(y)
        sp, vp = detectors.synthetic_probability(yp), detectors.voice_similarity(yp, enrolled_phone)
        print(f"{Path(p).name[:40]:40s} {len(y)/SR:4.1f}s | clean: synthetic {s:6.1%} voiceprint {v:5.2f} "
              f"| phone: synthetic {sp:6.1%} voiceprint {vp:5.2f}", flush=True)


if __name__ == "__main__":
    main()
