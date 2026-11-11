"""The two ML models: Spectra-AASIST (synthetic speech) and NVIDIA TitaNet (voiceprint).

Optional and heavy (torch, transformers, NeMo; install requirements-ml.txt). Everything is
imported lazily: without these packages the app runs on the scores these same models
produced ahead of time (scripts/score_calls.py), stored in each call file.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from functools import lru_cache

import numpy as np

from .audio import SR

# Spectra-AASIST: wav2vec2 XLS-R + AASIST anti-spoofing model (Apache-2.0). Its model.py is third-party
# code that runs locally, so it is pinned to the revision that was reviewed.
SPECTRA_REPO = "lab260/Spectra-AASIST"
SPECTRA_REV = "eb65c2662d9e646d72557b3f4bdd08b000068c7f"
SPECTRA_WIN = 64600              # ~4 s window, the length the model was trained on
SPECTRA_THRESHOLD = -1.140625    # model card: a bona fide logit below this means spoof (= probability 0.5 here)

TITANET = "nvidia/speakerverification_en_titanet_large"   # CC-BY-4.0, 192-dim speaker embeddings, 16 kHz mono

MODEL_NAMES = {"synthetic_voice": "Spectra-AASIST (lab260, rev eb65c26)",
               "voiceprint": "NVIDIA TitaNet-Large (speakerverification_en_titanet_large)"}

_cache: dict[str, object] = {}   # results by audio hash: a growing call only re-runs the new audio


def available() -> bool:
    """True when the ML packages are installed (does not load anything)."""
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers", "nemo", "huggingface_hub"))


@lru_cache(maxsize=1)
def device() -> str:
    """VSHIELD_DEVICE overrides; otherwise CUDA, then Apple-silicon MPS, then CPU."""
    import torch
    if os.getenv("VSHIELD_DEVICE"):
        return os.environ["VSHIELD_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _key(tag: str, y: np.ndarray) -> str:
    return tag + hashlib.sha1(np.ascontiguousarray(y, dtype=np.float32).tobytes()).hexdigest()


# ---- synthetic speech ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _spectra():
    import torch
    from huggingface_hub import hf_hub_download
    code = hf_hub_download(SPECTRA_REPO, "model.py", revision=SPECTRA_REV)
    spec = importlib.util.spec_from_file_location("spectra_aasist_model", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.SpectraAASIST.from_pretrained(SPECTRA_REPO, revision=SPECTRA_REV).eval().to(device())
    return model, torch


def _windows(y: np.ndarray) -> list[np.ndarray]:
    x = np.append(y[:1], y[1:] - 0.97 * y[:-1]).astype(np.float32)   # pre-emphasis, as in training
    if len(x) <= SPECTRA_WIN:
        return [np.resize(x, SPECTRA_WIN)]                             # repeat-pad short clips
    starts = list(range(0, len(x) - SPECTRA_WIN + 1, SPECTRA_WIN))
    if len(x) - (starts[-1] + SPECTRA_WIN) >= SR:                      # cover a tail of 1 s or more
        starts.append(len(x) - SPECTRA_WIN)
    return [x[s:s + SPECTRA_WIN] for s in starts]


def spectra_logits(y: np.ndarray) -> np.ndarray:
    """[windows, 2] logits (index 0 = spoof, 1 = bona fide) for 16 kHz mono audio."""
    model, torch = _spectra()
    wins = _windows(y)
    keys = [_key("spectra", w) for w in wins]
    todo = [(k, w) for k, w in zip(keys, wins) if k not in _cache]
    if todo:
        with torch.inference_mode():
            out = model(torch.from_numpy(np.stack([w for _, w in todo])).to(device())).float().cpu().numpy()
        _cache.update({k: o for (k, _), o in zip(todo, out)})
    return np.stack([_cache[k] for k in keys])


def synthetic_probability(y: np.ndarray) -> float:
    """0..1, where 0.5 is the model card's decision threshold."""
    bona = float(spectra_logits(y)[:, 1].mean())
    return float(1 / (1 + np.exp(-(SPECTRA_THRESHOLD - bona))))


# ---- voiceprint ------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _titanet():
    import nemo.collections.asr as nemo_asr
    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(TITANET, map_location=device()).eval()


def voice_embedding(y: np.ndarray) -> np.ndarray:
    k = _key("titanet", y)
    if k not in _cache:
        emb, _ = _titanet().infer_segment(y.astype(np.float32))
        _cache[k] = emb.squeeze().detach().cpu().numpy()
    return _cache[k]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def voice_similarity(y: np.ndarray, enrolled: np.ndarray) -> float:
    return cosine(voice_embedding(y), voice_embedding(enrolled))


def load() -> None:
    """Load both models up front (first call downloads ~2.6 GB of weights)."""
    _spectra()
    _titanet()
