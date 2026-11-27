"""Check Jegan's Spectra-AASIST + AASIST3 ensemble against Shield's current detector, on data it was NOT tuned on.

Jegan's pipeline (reference/spectra-aasist.ipynb), ported as written:
  overlapping 4.04 s windows (2.02 s hop) -> RMS silence gate on the raw audio -> pre-emphasis 0.97 per window ->
  Spectra-AASIST and Spectra-AASIST3 -> worst (minimum) bona fide logit of each -> synthetic if either is < 0.
Shield today (vshield/detectors.py): Spectra-AASIST only, back-to-back windows, MEAN bona fide logit, synthetic
if it is below the model card's -1.140625 (probability >= 0.5).

Sets: Jegan's 20 Voice samples (to reproduce his sheet) and our labelled eval sets (held out: test-clean; calibration:
dev-clean), each on clean audio and through the simulated 8 kHz G.711 phone line. Per-window logits are saved to
jegan_check_scores.json, so other aggregations can be compared without re-running the models.

    cd /Users/shivashankar/Downloads/deepfake-voice-shield
    .venv-ml/bin/python "/Users/shivashankar/Downloads/Shield own voices/jegan_ensemble_check.py" [samples test-clean dev-clean]
"""
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/shivashankar/Downloads/deepfake-voice-shield")
sys.path.insert(0, str(REPO))
from vshield import detectors  # noqa: E402
from vshield.audio import SR, load_wav, telephone  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_pavithra_voices import load as load_any  # noqa: E402

SAMPLES = Path("/Users/shivashankar/Downloads/Voice samples")
EVAL = REPO / "data" / "external" / "eval"
OUT = HERE / "jegan_check_scores.json"
V3_REPO, V3_REV = "lab260/Spectra-AASIST3", "bc0ded888080ddad493177bb53aa6f5b95219d7c"   # code reviewed 2026-11-27

WINDOW_LEN, HOP_LEN, SUBFRAME_LEN = 64600, 32300, 400      # Jegan's constants
RMS_THRESHOLD, MIN_ACTIVE_RATIO, PREEMPHASIS = 0.008, 0.25, 0.97
REAL_WORDS = ("genuine", "original", "human rep")


def load_v3():
    import torch
    from huggingface_hub import snapshot_download
    d = snapshot_download(V3_REPO, revision=V3_REV)
    spec = importlib.util.spec_from_file_location("spectra_aasist3_model", Path(d) / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SpectraAASIST3.from_pretrained(d).eval().to(detectors.device()), torch


def pad_or_crop(x: np.ndarray) -> np.ndarray:
    if len(x) >= WINDOW_LEN:
        s = (len(x) - WINDOW_LEN) // 2
        return x[s:s + WINDOW_LEN]
    return np.tile(x, WINDOW_LEN // len(x) + 1)[:WINDOW_LEN]


def jegan_windows(y: np.ndarray) -> list[np.ndarray]:
    """Overlapping windows that pass the RMS gate (on raw audio); the loudest one if none pass."""
    if len(y) <= WINDOW_LEN:
        return [pad_or_crop(y)]
    starts = range(0, len(y) - WINDOW_LEN + 1, HOP_LEN)
    keep = []
    for s in starts:
        c = y[s:s + WINDOW_LEN]
        if np.sqrt(np.mean(c ** 2)) < RMS_THRESHOLD:
            continue
        frames = c[: len(c) // SUBFRAME_LEN * SUBFRAME_LEN].reshape(-1, SUBFRAME_LEN)
        if np.mean(np.sqrt(np.mean(frames ** 2, axis=1)) > RMS_THRESHOLD) >= MIN_ACTIVE_RATIO:
            keep.append(c)
    return keep or [max((y[s:s + WINDOW_LEN] for s in starts), key=lambda c: float(np.mean(c ** 2)))]


def preemph(w: np.ndarray) -> np.ndarray:
    """torchaudio.functional.preemphasis: y[0] = x[0], y[t] = x[t] - 0.97 x[t-1]."""
    return np.append(w[:1], w[1:] - PREEMPHASIS * w[:-1]).astype(np.float32)


def bona_logits(model, torch, wins: list[np.ndarray], batch: int = 4, emph: bool = True) -> list[float]:
    """Bona fide logit per window, a few windows at a time (two 300M models on Apple-silicon memory)."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(wins), batch):
            x = np.stack([preemph(w) if emph else w for w in wins[i:i + batch]])
            x = torch.from_numpy(x.astype(np.float32)).to(detectors.device())
            out += [round(float(v), 4) for v in model(x).float().cpu().numpy()[:, 1]]
    if detectors.device() == "mps":
        torch.mps.empty_cache()
    return out


def shield_today(v1, torch, y: np.ndarray) -> float:
    """detectors.synthetic_probability, computed in small batches: same windows (pre-emphasis on the whole clip,
    back-to-back), mean bona fide logit, model-card threshold."""
    bona = float(np.mean(bona_logits(v1, torch, detectors._windows(y), emph=False)))
    return float(1 / (1 + np.exp(-(detectors.SPECTRA_THRESHOLD - bona))))


def clip_sets(names: list[str]) -> dict[str, list[dict]]:
    sets = {}
    if "samples" in names:
        sets["samples"] = [{"id": p.name, "label": "bonafide" if any(w in p.name.lower() for w in REAL_WORDS) else "spoof",
                            "attack": p.name.split()[0].lower() if not any(w in p.name.lower() for w in REAL_WORDS) else "none",
                            "path": p} for p in sorted(SAMPLES.iterdir()) if p.is_file() and not p.name.startswith(".")]
    for split in ("test-clean", "dev-clean"):
        if split in names:
            rows = [r for r in csv.DictReader(open(EVAL / split / "manifest.csv")) if r["role"] == "test"]
            sets[split] = [{"id": r["path"], "label": r["label"], "attack": r["attack"], "path": EVAL / split / r["path"]}
                           for r in rows]
    return sets


def main() -> None:
    names = sys.argv[1:] or ["samples", "test-clean", "dev-clean"]
    print("Loading Spectra-AASIST (pinned), Spectra-AASIST3 (pinned) ...", flush=True)
    v1, torch = detectors._spectra()
    v3, _ = load_v3()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for set_name, clips in clip_sets(names).items():
        for ch, fn in (("clean", lambda y: y), ("telephone", telephone)):
            key = f"{set_name}/{ch}"
            done = {r["id"] for r in results.get(key, [])}
            rows = [r for r in results.get(key, [])]
            t0 = time.time()
            for c in clips:
                if c["id"] in done:
                    continue
                y = load_any(c["path"]) if set_name == "samples" else load_wav(str(c["path"]))[0]
                y = fn(np.asarray(y, dtype=np.float32)).astype(np.float32)
                wins = jegan_windows(y)
                rows.append({"id": c["id"], "label": c["label"], "attack": c["attack"], "seconds": round(len(y) / SR, 2),
                             "ours_p": round(shield_today(v1, torch, y), 4),
                             "v1": bona_logits(v1, torch, wins), "v3": bona_logits(v3, torch, wins)})
                results[key] = rows
                OUT.write_text(json.dumps(results))                     # resumable: saved after every clip
            results[key] = rows
            OUT.write_text(json.dumps(results))
            print(f"{key}: {len(rows)} clips scored ({time.time() - t0:.0f} s)", flush=True)
    print("saved", OUT)


if __name__ == "__main__":
    main()
