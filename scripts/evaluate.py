"""Measure both models on the labelled evaluation sets (run in .venv-ml).

    .venv-ml/bin/python scripts/evaluate.py          # -> reports/eval.json (read by the Model evaluation page)

Sets come from scripts/build_eval_set.py: dev-clean (calibration split) and test-clean (held out; the split the
friend's TitaNet notebook used). Every test runs twice: on clean 16 kHz audio, and through a simulated phone line
(8 kHz G.711). Decision points are fixed in advance: synthetic probability 0.5 (the model card's threshold) and
voiceprint match at 0.70 (calibrated on dev-clean only).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vshield import detectors  # noqa: E402
from vshield.audio import SR, load_wav, telephone  # noqa: E402
from vshield.signals.voiceprint import THRESHOLDS  # noqa: E402

EVAL = ROOT / "data" / "external" / "eval"
CHANNELS = {"clean": lambda y: y, "telephone": telephone}
MATCH = THRESHOLDS[1]


def eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold; higher score = more 'positive'."""
    t = np.unique(np.r_[pos, neg])
    frr = np.searchsorted(np.sort(pos), t, side="left") / len(pos)        # positives below t
    far = 1 - np.searchsorted(np.sort(neg), t, side="left") / len(neg)    # negatives at or above t
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(t[i])


def r3(xs) -> list[float]:
    return [round(float(x), 3) for x in xs]


def evaluate_split(split: str) -> dict:
    rows = list(csv.DictReader(open(EVAL / split / "manifest.csv")))
    audio = {r["path"]: load_wav(str(EVAL / split / r["path"]))[0] for r in rows}
    tests = [r for r in rows if r["role"] == "test"]
    attacks = sorted({r["attack"] for r in tests if r["label"] == "spoof"})
    speakers = sorted({r["speaker"] for r in rows if r["role"] == "enroll"})
    out = {"counts": {"speakers": len(speakers), "real_test_clips": sum(r["label"] == "bonafide" for r in tests),
                      "fake_test_clips": sum(r["label"] == "spoof" for r in tests),
                      **{f"{a}_clips": sum(r["attack"] == a for r in tests) for a in attacks}},
           "channels": {}}
    for ch, fn in CHANNELS.items():
        # synthetic voice
        t0, secs, p = time.time(), 0.0, {}
        for r in tests:
            y = fn(audio[r["path"]])
            p[r["path"]] = detectors.synthetic_probability(y)
            secs += len(y) / SR
        rtf = (time.time() - t0) / secs
        real = np.array([p[r["path"]] for r in tests if r["label"] == "bonafide"])
        fake = np.array([p[r["path"]] for r in tests if r["label"] == "spoof"])
        e, thr = eer(fake, real)
        sv = {"eer": e, "eer_threshold": thr, "false_alarm_rate": float(np.mean(real > 0.5)),
              "detection_rate": {a: float(np.mean([p[r["path"]] > 0.5 for r in tests if r["attack"] == a])) for a in attacks},
              "seconds_per_audio_second": round(rtf, 4),
              "scores": {"real": r3(real), **{a: r3([p[r["path"]] for r in tests if r["attack"] == a]) for a in attacks}}}
        # voiceprint
        enr = {s: detectors.voice_embedding(fn(np.concatenate([audio[r["path"]] for r in rows
                                                                if r["role"] == "enroll" and r["speaker"] == s])))
               for s in speakers}
        same, other, clone, fooled = [], [], [], []
        for r in tests:
            if r["label"] == "bonafide":
                e_t = detectors.voice_embedding(fn(audio[r["path"]]))
                for s in speakers:
                    (same if s == r["speaker"] else other).append(detectors.cosine(e_t, enr[s]))
            elif r["attack"] == "chatterbox_clone":
                sim = detectors.cosine(detectors.voice_embedding(fn(audio[r["path"]])), enr[r["target"]])
                clone.append(sim)
                if sim >= MATCH:
                    fooled.append(p[r["path"]] > 0.5)
        same, other, clone = map(np.array, (same, other, clone))
        e, thr = eer(same, other)
        vp = {"eer": e, "eer_threshold": thr,
              "accept_rate": {"same_speaker": float(np.mean(same >= MATCH)), "other_speaker": float(np.mean(other >= MATCH)),
                              "clone": float(np.mean(clone >= MATCH))},
              "scores": {"same_speaker": r3(same), "other_speaker": r3(other), "clone": r3(clone)}}
        out["channels"][ch] = {
            "synthetic_voice": sv, "voiceprint": vp,
            "clones_fooling_voiceprint": len(fooled), "of_those_flagged_synthetic": int(np.sum(fooled))}
        print(f"  {split:10s} {ch:9s} spoof EER {sv['eer']:5.1%}  false alarms {sv['false_alarm_rate']:5.1%}  "
              f"caught {', '.join(f'{a} {v:.0%}' for a, v in sv['detection_rate'].items())}  |  voiceprint EER {e:5.1%}"
              f"  clones accepted {vp['accept_rate']['clone']:.0%}  |  {np.sum(fooled)}/{len(fooled)} of those flagged")
    return out


def main() -> None:
    splits = [s for s in ("dev-clean", "test-clean") if (EVAL / s / "manifest.csv").exists()]
    if not splits:
        sys.exit("No evaluation sets - run scripts/build_eval_set.py first")
    detectors.load()
    report = {"generated": date.today().isoformat(), "models": detectors.MODEL_NAMES,
              "decision_points": {"synthetic_voice": 0.5, "voiceprint_match": MATCH},
              "splits": {s: evaluate_split(s) for s in splits}}
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "eval.json").write_text(json.dumps(report) + "\n")
    print(f"-> {out / 'eval.json'}")


if __name__ == "__main__":
    main()
