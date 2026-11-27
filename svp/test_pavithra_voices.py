"""Test Pavithra's original voice against her AI clones with Shield's two real models.

  1. Voiceprint on file : NVIDIA TitaNet-Large embedding of her enrollment recording.
  2. Each clip         : Spectra-AASIST (is it synthetic?) + TitaNet similarity to her voiceprint (is it her?),
                         on clean audio and through a simulated 8 kHz phone line.
  3. Verdict           : what Shield concludes from the voice alone, with the demo's thresholds
                         (synthetic >= 0.5; voiceprint match >= 0.70, mismatch <= 0.55).

Standalone: reads the project's model wrappers, changes nothing. Defaults to Pavithra's files; override with
--enroll / --real / --clone. Run from the repo with the models environment:

    cd /Users/shivashankar/Downloads/deepfake-voice-shield
    .venv-ml/bin/python "/Users/shivashankar/Downloads/Shield own voices/test_pavithra_voices.py"
    .venv-ml/bin/python ".../test_pavithra_voices.py" --real a.m4a --clone b.mp3 c.wav     # any other pair
"""
import argparse
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path("/Users/shivashankar/Downloads/deepfake-voice-shield")
sys.path.insert(0, str(REPO))
from vshield import detectors  # noqa: E402
from vshield.audio import SR, resample, telephone  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLES = Path("/Users/shivashankar/Downloads/Voice samples")
DEFAULT_ENROLL = SAMPLES / "genuine voice - pavi.mp3"
DEFAULT_REAL = [SAMPLES / "Original voice - pavi.ogg"]
DEFAULT_CLONES = [SAMPLES / "pavis voice clone - more accurate.mp3", SAMPLES / "pavi_s voice clone.ogg",
                  HERE / "work" / "sun_clone_seed7.wav"]
SYNTH_AT, MATCH_AT, MISMATCH_AT = 0.5, 0.70, 0.55


def load(path: Path) -> np.ndarray:
    """16 kHz mono. Reads mp3/ogg/wav/flac (and m4a via macOS afconvert); keeps the part before a damaged tail."""
    if path.suffix.lower() in (".m4a", ".aac", ".mp4", ".caf"):
        tmp = Path(tempfile.mkdtemp()) / "clip.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(tmp)], check=True)
        path = tmp
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
    return resample(np.concatenate(blocks).mean(axis=1), sr, SR)


def verdict(p: float, sim: float) -> str:
    if p >= SYNTH_AT:
        return "ESCALATE: AI voice" + (" (and it fooled the voiceprint)" if sim >= MATCH_AT else "")
    if sim >= MATCH_AT:
        return "PROCEED: real human, matches her voiceprint"
    if sim <= MISMATCH_AT:
        return "STEP UP: human-sounding, but NOT her voiceprint"
    return "STEP UP: human, voiceprint not confirmed (one quick check)"


def score(y: np.ndarray, enrolled: np.ndarray) -> tuple[float, float]:
    return detectors.synthetic_probability(y), detectors.voice_similarity(y, enrolled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--enroll", type=Path, default=DEFAULT_ENROLL, help="her voiceprint recording (a separate recording)")
    ap.add_argument("--real", type=Path, nargs="+", default=DEFAULT_REAL, help="clips of her real voice")
    ap.add_argument("--clone", type=Path, nargs="+", default=[c for c in DEFAULT_CLONES if c.exists()], help="AI clones")
    args = ap.parse_args()

    print("Loading NVIDIA TitaNet-Large and Spectra-AASIST ...", flush=True)
    detectors.load()
    enrolled = load(args.enroll)
    half = len(enrolled) // 2
    same = detectors.cosine(detectors.voice_embedding(enrolled[:half]), detectors.voice_embedding(enrolled[half:]))
    lines = [f"# Pavithra: original voice vs AI clones ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Voiceprint on file: `{args.enroll.name}` ({len(enrolled) / SR:.0f} s). Sanity check: its two halves match "
             f"at {same:.2f} (same person, same recording).", "",
             "| Clip | Truth | Clean: synthetic / voiceprint | Phone line: synthetic / voiceprint | Shield (phone line) | Right? |",
             "|---|---|---|---|---|---|"]
    ok_real, stopped = [], []
    for truth, clips in (("real", args.real), ("AI clone", args.clone)):
        for clip in clips:
            y = load(clip)
            p, sim = score(y, enrolled)
            pp, simp = score(telephone(y), telephone(enrolled))
            v = verdict(pp, simp)
            right = v.startswith("PROCEED") if truth == "real" else not v.startswith("PROCEED")
            (ok_real if truth == "real" else stopped).append(right)
            lines.append(f"| {clip.name} | {truth} | {p:.0%} / {sim:.2f} | {pp:.0%} / {simp:.2f} | {v} | "
                         f"{'yes' if right else 'NO'} |")
    lines += ["", f"- Real voice accepted: {sum(ok_real)} of {len(ok_real)}",
              f"- AI clones stopped (not allowed to proceed): {sum(stopped)} of {len(stopped)}",
              "- A clone that is human-sounding but not her voiceprint is stopped by the voiceprint; a clone the detector "
              "hears as synthetic is stopped by the deepfake detector. Shield needs only one layer to catch it.",
              f"- Thresholds: synthetic >= {SYNTH_AT}; voiceprint match >= {MATCH_AT}, mismatch <= {MISMATCH_AT}. Small "
              "sample: a demonstration, not an accuracy measurement."]
    report = "\n".join(lines)
    print("\n" + report)
    out = HERE / "pavithra_test_report.md"
    out.write_text(report + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
