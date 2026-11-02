"""Build the spike evaluation set + real-audio demo calls (run in the generator venv).

    .venv-gen/bin/python scripts/build_sample_set.py [--clone-targets 8]

Bona fide speech: LibriSpeech dev-clean (CC-BY-4.0), downloaded to data/external/.
Spoofed speech, all generated locally:
  kokoro            Kokoro-82M stock TTS voices (Apache-2.0)                  -> "AI bot" attack
  chatterbox_clone  Chatterbox zero-shot clones of LibriSpeech speakers (MIT)  -> "voice clone" attack
  macos_say         macOS `say` voices (older concatenative/parametric TTS)

Writes (git-ignored, nothing here is redistributed):
  data/samples/eval/manifest.csv   path,label,attack,speaker,target,role,text
  data/samples/demo/               accounts.json, enrollment/, audio/, calls/ - same three stories
                                   as data/, but with real speech; open with VSHIELD_DATA=data/samples/demo
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vshield.dsp import SR, save_wav  # noqa: E402
import generate_demo_data as demo  # noqa: E402

LIBRI = ROOT / "data" / "external" / "librispeech" / "LibriSpeech"
OUT = ROOT / "data" / "samples"
KOKORO_VOICES = ["af_heart", "af_bella", "am_michael", "am_fenrir", "bf_emma", "bm_george", "af_nicole", "am_puck"]
SAY_VOICES = ["Samantha", "Daniel", "Karen", "Rishi", "Moira", "Fred"]
CALL_LINES = [text for spec in demo.CALLS.values() for role, text, _ in spec["script"] if role == "caller"]


def to_sr(y: np.ndarray, sr: int) -> np.ndarray:
    y = y.astype(np.float32)
    if sr != SR:
        g = np.gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return 0.9 * y / (np.abs(y).max() + 1e-9)


def librispeech() -> dict[str, list[dict]]:
    """speaker -> utterances (4-15 s), each {path, text, dur}."""
    gender = {}
    for line in (LIBRI / "SPEAKERS.TXT").read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[2] == "dev-clean":
            gender[parts[0]] = parts[1]
    spk = {}
    for trans in sorted((LIBRI / "dev-clean").glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            dur = sf.info(str(path)).duration
            if 4.0 <= dur <= 15.0:
                spk.setdefault(uid.split("-")[0], []).append({"path": path, "text": text.capitalize() + ".", "dur": dur})
    return {s: u for s, u in spk.items() if len(u) >= 8}, gender


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-targets", type=int, default=8)
    args = ap.parse_args()

    import torch
    from chatterbox.tts import ChatterboxTTS
    from kokoro import KPipeline

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ev, dm = OUT / "eval", OUT / "demo"
    for d in (ev / "bonafide", ev / "enroll", ev / "kokoro", ev / "chatterbox_clone", ev / "macos_say",
              dm / "enrollment", dm / "audio", dm / "calls"):
        d.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(y, rel, label, attack, speaker, target, role, text):
        if not (ev / rel).exists():
            save_wav(str(ev / rel), y() if callable(y) else y)
        rows.append({"path": f"eval/{rel}", "label": label, "attack": attack, "speaker": speaker,
                     "target": target, "role": role, "text": text})

    # ---- bona fide: per speaker 2 enrollment, 3 test, 1 clone reference -------------
    spk, gender = librispeech()
    speakers = sorted(spk)
    print(f"{len(speakers)} LibriSpeech speakers")
    plan = {}
    for s in speakers:
        u = sorted(spk[s], key=lambda x: x["path"].name)
        plan[s] = {"enroll": u[0:2], "test": u[2:5], "ref": max(u[5:], key=lambda x: x["dur"]), "spare": u[5:]}
        for i, x in enumerate(plan[s]["enroll"]):
            add(to_sr(*sf.read(str(x["path"]))), f"enroll/{s}_{i}.wav", "bonafide", "none", s, s, "enroll", x["text"])
        for i, x in enumerate(plan[s]["test"]):
            add(to_sr(*sf.read(str(x["path"]))), f"bonafide/{s}_{i}.wav", "bonafide", "none", s, s, "test", x["text"])

    f_spk = [s for s in speakers if gender.get(s) == "F"]
    m_spk = [s for s in speakers if gender.get(s) == "M"]
    half = args.clone_targets // 2
    targets = f_spk[:half] + m_spk[:args.clone_targets - half]

    # ---- Chatterbox zero-shot clones of target speakers ------------------------------
    print(f"Chatterbox on {dev}: cloning {len(targets)} speakers")
    cb = ChatterboxTTS.from_pretrained(device=dev)
    for ti, s in enumerate(targets):
        texts = [CALL_LINES[(2 * ti) % len(CALL_LINES)], CALL_LINES[(2 * ti + 1) % len(CALL_LINES)],
                 plan[s]["test"][0]["text"], plan[s]["test"][1]["text"]]   # 2 call lines + 2 content-matched
        for j, text in enumerate(texts):   # generated lazily: clips already on disk are reused
            gen = lambda text=text: to_sr(cb.generate(text, audio_prompt_path=str(plan[s]["ref"]["path"]))
                                          .squeeze(0).cpu().numpy(), cb.sr)
            add(gen, f"chatterbox_clone/{s}_{j}.wav", "spoof", "chatterbox_clone", f"clone:{s}", s, "test", text)
        print(f"  cloned {s}")

    # ---- Kokoro stock voices -----------------------------------------------------------
    kp = {"a": KPipeline(lang_code="a"), "b": KPipeline(lang_code="b")}

    def kokoro(text, voice):
        audio = [a for _, _, a in kp[voice[0]](text, voice=voice)]
        return to_sr(np.concatenate([np.asarray(a) for a in audio]), 24_000)

    for vi, v in enumerate(KOKORO_VOICES):
        lib = plan[speakers[vi]]["test"]
        texts = [CALL_LINES[(2 * vi) % len(CALL_LINES)], CALL_LINES[(2 * vi + 1) % len(CALL_LINES)], lib[0]["text"], lib[1]["text"]]
        for j, text in enumerate(texts):
            add(kokoro(text, v), f"kokoro/{v}_{j}.wav", "spoof", "kokoro", f"kokoro:{v}", "", "test", text)
    print("Kokoro done")

    # ---- macOS say ---------------------------------------------------------------------
    tmp = ev / "macos_say" / "_tmp.wav"
    for vi, v in enumerate(SAY_VOICES):
        for j in range(3):
            text = CALL_LINES[(3 * vi + j) % len(CALL_LINES)]
            subprocess.run(["say", "-v", v, "--data-format=LEI16@16000", "-o", str(tmp), text], check=True)
            y, sr = sf.read(str(tmp))
            add(to_sr(y, sr), f"macos_say/{v}_{j}.wav", "spoof", "macos_say", f"say:{v}", "", "test", text)
    tmp.unlink()
    print("macOS say done")

    with open(ev / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips -> {ev / 'manifest.csv'}")

    # ---- real-audio demo calls: same stories, real speech ------------------------------
    # Jordan = a real speaker (genuine call uses their real utterances);
    # Alex   = a clone target (clone call = Chatterbox clone of Alex speaking the SIM-swap script);
    # Sam    = another real speaker (bot call = Kokoro stock voice speaking the negotiation script).
    # Cast the best-recorded speakers: LibriSpeech is home-recorded audiobooks and some microphones are
    # muffled; a clone copies that channel too, so it would sound poor in the demo.
    def brightness(s):
        y, sr = sf.read(str(plan[s]["ref"]["path"]))
        f, p = welch(y, sr)
        return p[f > 2000].sum() / p.sum()

    jordan = max((s for s in m_spk if s not in targets), key=brightness)
    alex = max(targets, key=brightness)
    sam = max((s for s in f_spk if s not in targets), key=brightness)
    cast = {"ACC-1001": jordan, "ACC-1003": alex, "ACC-2002": sam}
    accounts = json.loads(json.dumps(demo.ACCOUNTS))
    for acc, s in cast.items():
        y = np.concatenate([to_sr(*sf.read(str(x["path"]))) for x in plan[s]["enroll"]])
        save_wav(str(dm / "enrollment" / f"{acc}.wav"), y)
        accounts[acc]["voiceprint"] = f"enrollment/{acc}.wav"
    (dm / "accounts.json").write_text(json.dumps(accounts, indent=2))

    genuine_utts = iter(plan[jordan]["test"] + plan[jordan]["spare"])

    def genuine_voice(text, i):
        x = next(genuine_utts)
        return to_sr(*sf.read(str(x["path"]))), x["text"]   # real speech, so the transcript is what was said

    def clone_voice(text, i):
        wav = cb.generate(text, audio_prompt_path=str(plan[alex]["ref"]["path"]))
        return to_sr(wav.squeeze(0).cpu().numpy(), cb.sr), text

    def bot_voice(text, i):
        return kokoro(text, "am_michael"), text

    for cid, voice in (("genuine_customer", genuine_voice), ("voice_clone", clone_voice), ("ai_negotiator", bot_voice)):
        call = demo.build_call(cid, demo.CALLS[cid], voice=voice, out=dm)
        call["title"] += " (real audio)"
        (dm / "calls" / f"{cid}.json").write_text(json.dumps(call, indent=2))
        print(f"  demo call {cid}: {call['turns'][-1]['end']:.1f}s")
    (dm / "cast.json").write_text(json.dumps({"librispeech_speakers": cast, "clone_targets": targets}, indent=2))
    print(f"Real-audio demo -> {dm}")


if __name__ == "__main__":
    main()
