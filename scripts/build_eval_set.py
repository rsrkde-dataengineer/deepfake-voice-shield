"""Build the labelled evaluation sets: real human speech vs. synthetic speech (run in .venv-gen).

    .venv-gen/bin/python scripts/build_eval_set.py                    # both splits, ~30 min each on an M1
    .venv-gen/bin/python scripts/build_eval_set.py --split test-clean

dev-clean   calibration split: the voiceprint thresholds were set here.
test-clean  held-out split (the one in the friend's TitaNet notebook): new speakers, new clone targets,
            and Kokoro / macOS voices that appear nowhere else.

Per split, in data/external/eval/<split>/ (git-ignored, not redistributed):
  enroll/            2 utterances per speaker: the enrolled voiceprint
  bonafide/          3 more utterances per speaker: real human test clips
  chatterbox_clone/  4 clips per clone target (Chatterbox, MIT), cloned from a separate reference utterance
  kokoro/            4 clips per Kokoro-82M stock voice (Apache-2.0)
  macos_say/         3 clips per macOS voice (older TTS)
  manifest.csv       path,label,attack,speaker,target,role,text
Clips already on disk are reused, so a rerun only generates what is missing.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vshield.audio import SR, resample, save_wav  # noqa: E402

LIBRI = ROOT / "data" / "external" / "librispeech" / "LibriSpeech"
OUT = ROOT / "data" / "external" / "eval"
VOICES = {
    "dev-clean": {"kokoro": ["af_heart", "af_bella", "am_michael", "am_fenrir", "bf_emma", "bm_george", "af_nicole", "am_puck"],
                  "say": ["Samantha", "Daniel", "Karen", "Rishi", "Moira", "Fred"]},
    "test-clean": {"kokoro": ["af_sarah", "af_sky", "am_adam", "am_eric", "bf_isabella", "bm_lewis", "af_river", "am_liam"],
                   "say": ["Tessa", "Aman", "Eddy (English (US))", "Flo (English (US))", "Reed (English (US))",
                           "Shelley (English (US))"]},
}
# Call-centre lines for the fake voices (half of the fake clips say these, half say LibriSpeech sentences).
CALL_LINES = [
    "Hi, um, yeah, I was looking at my bill and I think I got charged twice for, uh, the TravelPass?",
    "Sure, it's zero seven three zero two.",
    "Oh, let me think, I was in Toronto for like three days and I only turned it on once, I mean I think I only "
    "turned it on once, and then the second—",
    "Oh, yes, sorry, October, yeah.",
    "Oh great, thank you so much, that's, uh, really helpful.",
    "Hi Daniel. My phone was stolen this morning and I need to move my number to a new SIM card right away.",
    "Yes, it's one zero zero one six.",
    "I can't access that email right now. Can you send the code to a different email instead? I'm traveling and it's urgent.",
    "Silver maple eighteen.",
    "Sure. Please hurry, I really need my number working today.",
    "Hello Priya. This is Sam Rivera, the account holder. I have been a loyal customer for six years and I would like "
    "to review my monthly bill.",
    "My current bill is two hundred thirty one dollars. A competitor is offering a comparable plan for one hundred "
    "sixty dollars. I would like to discuss a retention discount.",
    "Thank you. Specifically, I am requesting a recurring monthly credit of twenty five dollars for twelve months, "
    "otherwise I will need to port my number to the competitor, and I would also like the activation fees waived "
    "on my next upgrade.",
    "Yes, all three lines are on my account. Can you confirm the credit amount?",
    "Green lantern sixty two.",
    "Thank you. I would appreciate the twenty five dollar monthly credit applied today.",
]


def to16k(y: np.ndarray, sr: int) -> np.ndarray:
    y = resample(np.asarray(y, dtype=np.float32), sr, SR)
    return 0.9 * y / (np.abs(y).max() + 1e-9)


def librispeech(split: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """speaker -> utterances of 4-15 s, each {path, text, dur}; plus speaker -> gender."""
    gender = {}
    for line in (LIBRI / "SPEAKERS.TXT").read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[2] == split:
            gender[parts[0]] = parts[1]
    spk: dict[str, list[dict]] = {}
    for trans in sorted((LIBRI / split).glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            dur = sf.info(str(path)).duration
            if 4.0 <= dur <= 15.0:
                spk.setdefault(uid.split("-")[0], []).append({"path": path, "text": text.capitalize() + ".", "dur": dur})
    return {s: u for s, u in spk.items() if len(u) >= 8}, gender


def build(split: str, n_targets: int) -> None:
    import torch
    from chatterbox.tts import ChatterboxTTS
    from kokoro import KPipeline

    out = OUT / split
    for d in ("enroll", "bonafide", "chatterbox_clone", "kokoro", "macos_say"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def add(y, rel, label, attack, speaker, target, role, text):
        if not (out / rel).exists():
            save_wav(str(out / rel), y() if callable(y) else y, SR)
        rows.append({"path": rel, "label": label, "attack": attack, "speaker": speaker, "target": target,
                     "role": role, "text": text})

    spk, gender = librispeech(split)
    speakers = sorted(spk)
    plan = {}
    for s in speakers:
        u = sorted(spk[s], key=lambda x: x["path"].name)
        plan[s] = {"enroll": u[0:2], "test": u[2:5], "ref": max(u[5:], key=lambda x: x["dur"])}
        for i, x in enumerate(plan[s]["enroll"]):
            add(lambda x=x: to16k(*sf.read(str(x["path"]))), f"enroll/{s}_{i}.wav", "bonafide", "none", s, s, "enroll", x["text"])
        for i, x in enumerate(plan[s]["test"]):
            add(lambda x=x: to16k(*sf.read(str(x["path"]))), f"bonafide/{s}_{i}.wav", "bonafide", "none", s, s, "test", x["text"])
    print(f"{split}: {len(speakers)} speakers")

    half = n_targets // 2
    targets = [s for s in speakers if gender.get(s) == "F"][:half] + [s for s in speakers if gender.get(s) == "M"][:n_targets - half]
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    cloner = None
    n = len(CALL_LINES)
    for ti, s in enumerate(targets):
        texts = [CALL_LINES[(2 * ti) % n], CALL_LINES[(2 * ti + 1) % n], plan[s]["test"][0]["text"], plan[s]["test"][1]["text"]]
        for j, text in enumerate(texts):
            def gen(text=text, s=s):
                nonlocal cloner
                cloner = cloner or ChatterboxTTS.from_pretrained(device=dev)
                wav = cloner.generate(text, audio_prompt_path=str(plan[s]["ref"]["path"]))
                return to16k(wav.squeeze(0).cpu().numpy(), cloner.sr)
            add(gen, f"chatterbox_clone/{s}_{j}.wav", "spoof", "chatterbox_clone", f"clone:{s}", s, "test", text)
        print(f"  clone target {s} done")

    kp = {"a": KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M"),
          "b": KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")}
    for vi, v in enumerate(VOICES[split]["kokoro"]):
        lib = plan[speakers[vi]]["test"]
        texts = [CALL_LINES[(2 * vi) % n], CALL_LINES[(2 * vi + 1) % n], lib[0]["text"], lib[1]["text"]]
        for j, text in enumerate(texts):
            gen = lambda text=text, v=v: to16k(np.concatenate([np.asarray(a) for _, _, a in kp[v[0]](text, voice=v)]), 24_000)
            add(gen, f"kokoro/{v}_{j}.wav", "spoof", "kokoro", f"kokoro:{v}", "", "test", text)

    tmp = out / "macos_say" / "_tmp.wav"
    for vi, v in enumerate(VOICES[split]["say"]):
        slug = v.split(" ")[0]
        for j in range(3):
            text = CALL_LINES[(3 * vi + j) % n]
            def gen(text=text, v=v):
                subprocess.run(["say", "-v", v, "--data-format=LEI16@16000", "-o", str(tmp), text], check=True)
                y, sr = sf.read(str(tmp))
                return to16k(y, sr)
            add(gen, f"macos_say/{slug}_{j}.wav", "spoof", "macos_say", f"say:{slug}", "", "test", text)
    tmp.unlink(missing_ok=True)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{split}: {len(rows)} clips -> {out / 'manifest.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(VOICES), action="append")
    ap.add_argument("--clone-targets", type=int, default=8)
    args = ap.parse_args()
    for split in args.split or list(VOICES):
        build(split, args.clone_targets)


if __name__ == "__main__":
    main()
