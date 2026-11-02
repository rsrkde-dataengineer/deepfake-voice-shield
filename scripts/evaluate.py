"""Evaluate the real backends on the sample set (roadmap item 3, spike version).

    .venv-ml/bin/python scripts/evaluate.py [--spoof-models spectra hf:<model_id> ...]

Needs data/samples/eval/manifest.csv from scripts/build_sample_set.py.
Every test runs twice: clean 16 kHz, and through a simulated phone line (8 kHz G.711).
Writes reports/spike_eval.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vshield.dsp import SR, load_wav, telephone  # noqa: E402
from vshield.signals import acoustic, voiceprint  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
CHANNELS = {"clean": lambda y: y, "telephone": telephone}


def eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold; higher score = more 'positive'."""
    best = (2.0, 0.0, 1.0)
    for t in np.unique(np.concatenate([pos, neg])):
        frr, far = np.mean(pos < t), np.mean(neg >= t)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), t, (far + frr) / 2)
    return float(best[2]), float(best[1])


def spoof_score(model: str, y: np.ndarray) -> float:
    """Probability-like synthetic score in [0, 1] (0.5 = the model's own decision point)."""
    if model == "spectra":
        return acoustic._spectra_score(y)[0]
    return acoustic._hf_score(y, model[3:])[0]


def eval_spoof(model: str, rows: list[dict], audio: dict) -> dict:
    out = {}
    for ch, fn in CHANNELS.items():
        scores, secs, t0 = {}, 0.0, time.time()
        for r in rows:
            y = fn(audio[r["path"]])
            scores[r["path"]] = spoof_score(model, y)
            secs += len(y) / SR
        rtf = (time.time() - t0) / secs
        bona = np.array([scores[r["path"]] for r in rows if r["label"] == "bonafide"])
        spoof = np.array([scores[r["path"]] for r in rows if r["label"] == "spoof"])
        e, thr = eer(spoof, bona)
        res = {"eer": e, "eer_threshold": thr, "false_alarm_at_0.5": float(np.mean(bona > 0.5)),
               "bonafide_p_max": float(bona.max()), "bonafide_p_p95": float(np.percentile(bona, 95)),
               "spoof_p_min": float(spoof.min()), "spoof_p_p5": float(np.percentile(spoof, 5)),
               "n_bonafide": len(bona), "real_time_factor": rtf, "detected_at_0.5": {}}
        for atk in sorted({r["attack"] for r in rows if r["label"] == "spoof"}):
            s = np.array([scores[r["path"]] for r in rows if r["attack"] == atk])
            res["detected_at_0.5"][atk] = {"rate": float(np.mean(s > 0.5)), "n": len(s)}
        out[ch] = res
        print(f"  {model:50s} {ch:9s} EER {e:6.1%}  false alarms {res['false_alarm_at_0.5']:5.1%}  "
              + "  ".join(f"{a} {v['rate']:.0%}" for a, v in res["detected_at_0.5"].items()) + f"  RTF {rtf:.3f}")
    return out


def eval_voiceprint(rows: list[dict], audio: dict) -> dict:
    lo, hi = voiceprint.THRESH["titanet"]
    out = {}
    enroll_rows = [r for r in rows if r["role"] == "enroll"]
    speakers = sorted({r["speaker"] for r in enroll_rows})
    tests = [r for r in rows if r["role"] == "test" and (r["label"] == "bonafide" or r["attack"] == "chatterbox_clone")]
    for ch, fn in CHANNELS.items():
        enr = {s: voiceprint._embed_titanet(fn(np.concatenate([audio[r["path"]] for r in enroll_rows if r["speaker"] == s])))
               for s in speakers}
        emb = {r["path"]: voiceprint._embed_titanet(fn(audio[r["path"]])) for r in tests}
        target, nontarget, clone = [], [], []
        for r in tests:
            for s in speakers:
                sim = voiceprint.cosine(emb[r["path"]], enr[s])
                if r["label"] == "bonafide":
                    (target if s == r["speaker"] else nontarget).append(sim)
                elif s == r["target"]:
                    clone.append(sim)
        target, nontarget, clone = map(np.array, (target, nontarget, clone))
        e, thr = eer(target, nontarget)
        res = {"eer": e, "eer_threshold": thr,
               "target_mean": float(target.mean()), "nontarget_mean": float(nontarget.mean()),
               "clone_mean": float(clone.mean()),
               "target_min": float(target.min()), "target_p5": float(np.percentile(target, 5)),
               "nontarget_max": float(nontarget.max()), "nontarget_p99": float(np.percentile(nontarget, 99)),
               "clone_min": float(clone.min()), "clone_max": float(clone.max()),
               "clone_accepted_at_eer_threshold": float(np.mean(clone >= thr)),
               "clone_accepted_at_0.70": float(np.mean(clone >= 0.70)),
               f"genuine_accepted_at_{hi}": float(np.mean(target >= hi)),
               f"impostor_accepted_at_{hi}": float(np.mean(nontarget >= hi)),
               f"clone_accepted_at_{hi}": float(np.mean(clone >= hi)),
               "n_target": len(target), "n_nontarget": len(nontarget), "n_clone": len(clone)}
        out[ch] = res
        print(f"  TitaNet {ch:9s} EER {e:5.1%} @ {thr:.2f} | cosine genuine {target.mean():.2f}  impostor "
              f"{nontarget.mean():.2f}  clone {clone.mean():.2f} | clones accepted @EER thr {res['clone_accepted_at_eer_threshold']:.0%}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spoof-models", nargs="+", default=["spectra", "hf:MelodyMachine/Deepfake-audio-detection-V2"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SAMPLES / "eval" / "manifest.csv")))
    audio = {r["path"]: load_wav(str(SAMPLES / r["path"]))[0] for r in rows}
    tests = [r for r in rows if r["role"] == "test"]
    print(f"{len(rows)} clips: {sum(r['label'] == 'bonafide' for r in tests)} bona fide test, "
          f"{sum(r['label'] == 'spoof' for r in tests)} spoof, {sum(r['role'] == 'enroll' for r in rows)} enrollment")

    # warm up (model download/load is not part of latency)
    acoustic._spectra(); voiceprint._titanet()
    report = {"anti_spoofing": {}, "voiceprint": {}}
    for m in args.spoof_models:
        report["anti_spoofing"][m] = eval_spoof(m, tests, audio)
    report["voiceprint"]["titanet"] = eval_voiceprint(rows, audio)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "spike_eval.json").write_text(json.dumps(report, indent=2))
    print(f"-> {out / 'spike_eval.json'}")


if __name__ == "__main__":
    main()
