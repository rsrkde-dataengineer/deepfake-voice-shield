"""Quick check, standalone: do Shield's two models tell Pavithra's real voice from her clone?

Uses the project's model wrappers (vshield.detectors) read-only; writes nothing to the project.
Enrolled voiceprint = 'genuine voice - pavi.mp3' (31 s), like a customer's voice on file.
Run: .venv-ml/bin/python own_voice_check.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path("/Users/shivashankar/Downloads/deepfake-voice-shield")
sys.path.insert(0, str(REPO))
from vshield import detectors  # noqa: E402
from vshield.audio import SR, resample, telephone  # noqa: E402

DIR = Path("/Users/shivashankar/Downloads/Voice samples")
ENROLL = "genuine voice - pavi.mp3"
FOCUS = [("Original voice - pavi.ogg", "Pavithra: real voice"),
         ("pavis voice clone - more accurate.mp3", "Pavithra: AI clone (accurate)"),
         ("pavi_s voice clone.ogg", "Pavithra: AI clone (earlier)")]
SYNTH_AT, MATCH_AT, MISMATCH_AT = 0.5, 0.70, 0.55


def load(name: str) -> np.ndarray:
    """Decode in 1 s blocks and stop at a damaged tail (one MP3 here has bad bytes at its very end)."""
    blocks = []
    with sf.SoundFile(DIR / name) as f:
        sr = f.samplerate
        while True:
            try:
                b = f.read(sr, dtype="float32", always_2d=True)
            except sf.LibsndfileError:
                print(f"  (note: '{name}' is damaged after {sum(len(x) for x in blocks) / sr:.1f} s; using the part before)")
                break
            if not len(b):
                break
            blocks.append(b)
    return resample(np.concatenate(blocks).mean(axis=1), sr, SR)


def verdict(p: float, sim: float) -> str:
    voice = "AI" if p >= SYNTH_AT else "human"
    who = "matches Pavithra" if sim >= MATCH_AT else "NOT Pavithra" if sim <= MISMATCH_AT else "unsure"
    return f"{voice}, {who}"


def row(name: str, label: str, enrolled: np.ndarray, enrolled_phone: np.ndarray) -> None:
    y = load(name)
    p, sim = detectors.synthetic_probability(y), detectors.voice_similarity(y, enrolled)
    yp = telephone(y)
    pp, simp = detectors.synthetic_probability(yp), detectors.voice_similarity(yp, enrolled_phone)
    print(f"{label:34s} {len(y) / SR:5.1f}s | clean: synthetic {p:5.1%}  voiceprint {sim:5.2f}  -> {verdict(p, sim):24s}"
          f"| phone line: synthetic {pp:5.1%}  voiceprint {simp:5.2f}  -> {verdict(pp, simp)}")


def main() -> None:
    print("loading models ...", flush=True)
    detectors.load()
    enrolled = load(ENROLL)
    half = len(enrolled) // 2
    print(f"\nEnrolled voiceprint: '{ENROLL}' ({len(enrolled) / SR:.1f} s)")
    print(f"Sanity: first half vs second half of the same recording: "
          f"{detectors.cosine(detectors.voice_embedding(enrolled[:half]), detectors.voice_embedding(enrolled[half:])):.2f}"
          f"   (match at >= {MATCH_AT})\n")
    enrolled_phone = telephone(enrolled)
    print("THE GAME PAIR")
    for name, label in FOCUS:
        row(name, label, enrolled, enrolled_phone)
    print("\nEVERYTHING ELSE IN THE FOLDER (voiceprint compared with Pavithra)")
    focus = {n for n, _ in FOCUS} | {ENROLL}
    for f in sorted(DIR.iterdir()):
        if f.name not in focus and f.suffix.lower() in (".mp3", ".ogg", ".wav", ".m4a"):
            row(f.name, f.stem[:34], enrolled, enrolled_phone)


if __name__ == "__main__":
    main()
