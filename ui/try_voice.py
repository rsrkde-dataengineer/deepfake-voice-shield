"""Page 2 - Try your own voice: record, upload or pick a clip and run both models on it."""
from __future__ import annotations

import io
import time

import numpy as np
import streamlit as st

from ui import style as S
from vshield import detectors, fusion
from vshield.audio import SR, load_wav, resample, telephone, wav_bytes
from vshield.session import DATA, list_calls, load_accounts, load_call
from vshield.signals import synthetic_voice, voiceprint

MAX_SECONDS = 30
ME = "__me__"


@st.cache_resource(show_spinner="Loading Spectra-AASIST and NVIDIA TitaNet (the first run downloads ~2.6 GB)...")
def _load_models() -> bool:
    detectors.load()
    return True


def _decode(data: bytes) -> np.ndarray:
    import soundfile as sf   # installed with requirements-ml.txt
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    y = y.mean(axis=1) if y.ndim > 1 else y
    return resample(y, sr, SR)[: MAX_SECONDS * SR]


def _demo_clips() -> dict[str, dict]:
    clips = {}
    for cid in list_calls():
        call = load_call(cid)
        last = [t for t in call["turns"] if t["speaker"] == "caller"][-1]
        clips[cid] = {"label": f"{S.CALL_ICONS.get(cid, ':material/call:')} {call['title']}", "account": call["account_id"],
                      "path": DATA / call["audio"]["caller"], "stored": last["scores"], "truth": call["ground_truth"]}
    return clips


def render() -> None:
    S.header("Try <span>your own voice</span>",
             "Record or upload speech and run both models on it. Compare it with a customer's enrolled voiceprint, "
             "or enroll your own voice first.")
    accounts = load_accounts()
    clips = _demo_clips()
    live = detectors.available()
    if not live:
        st.info("The models are not installed on this machine, so only the demo clips can be checked here, using the "
                "scores the models produced ahead of time. To record or upload your own audio, install "
                "`requirements-ml.txt` and start the app from that environment (see README).", icon=":material/info:")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("##### 1 · The caller's voice")
        sources = ["Demo clip", "Record", "Upload"] if live else ["Demo clip"]
        source = st.segmented_control("Source", sources, default="Demo clip", label_visibility="collapsed")
        audio, claimed, stored, truth = None, None, None, None
        if source == "Demo clip":
            key = st.selectbox("Clip", list(clips), format_func=lambda k: clips[k]["label"])
            clip = clips[key]
            audio, _ = load_wav(str(clip["path"]))
            claimed, stored, truth = clip["account"], clip["stored"], clip["truth"]
            st.caption(f"The caller side of this call ({len(audio) / SR:.0f} s of speech). Ground truth: {truth}.")
        elif source == "Record":
            rec = st.audio_input("Speak for 5-20 seconds", sample_rate=SR)
            audio = _decode(rec.getvalue()) if rec else None
        elif source == "Upload":
            up = st.file_uploader("WAV, FLAC or OGG", type=["wav", "flac", "ogg"])
            audio = _decode(up.getvalue()) if up else None
        phone = st.toggle("Send it through a phone line (8 kHz G.711)", value=False, disabled=stored is not None,
                          help="Demo clips use their stored scores, which were computed on the clean audio.")

    with c2:
        st.markdown("##### 2 · Who the caller claims to be")
        options = list(accounts) + ([ME] if live else [])
        idx = options.index(claimed) if claimed in options else 0
        who = st.selectbox("Enrolled voice", options, index=idx, disabled=stored is not None,
                           format_func=lambda a: "Me - enroll my voice now" if a == ME else
                           f"{accounts[a]['name']} ({a})")
        enrolled = None
        if who == ME:
            rec = st.audio_input("Enroll: read anything aloud for 10-20 seconds", sample_rate=SR, key="enroll")
            up = st.file_uploader("...or upload a recording of yourself", type=["wav", "flac", "ogg"], key="enroll_file",
                                  help="Browsers only allow the microphone on localhost or HTTPS.")
            if rec or up:
                st.session_state.my_voice = _decode((rec or up).getvalue())
            enrolled = st.session_state.get("my_voice")
            if enrolled is None:
                st.caption("Record or upload your enrollment first.")
            else:
                st.caption(f"Enrolled: {len(enrolled) / SR:.0f} s of your voice.")
        elif who:
            enrolled, _ = load_wav(str(DATA / accounts[who]["voiceprint"]))
            st.caption(f"Enrolled voiceprint: {accounts[who]['voiceprint_source']}")

    if audio is None or len(audio) < SR:
        st.caption("Pick a demo clip, or record or upload at least a second of speech.")
        return
    if phone:
        audio = telephone(audio)
    st.audio(wav_bytes(audio, SR))
    if not st.button("Analyse", type="primary", icon=":material/graphic_eq:"):
        return

    secs = len(audio) / SR
    t0 = time.time()
    if stored is not None:
        p, sim = stored["synthetic_voice"], stored["voice_similarity"]
        how = "Stored scores from Spectra-AASIST and NVIDIA TitaNet for this exact audio."
    else:
        if enrolled is None:
            st.warning("Enroll a voice first.")
            return
        _load_models()
        with st.spinner("Running Spectra-AASIST and NVIDIA TitaNet..."):
            p = detectors.synthetic_probability(audio)
            if phone:
                enrolled = telephone(enrolled)
            sim = detectors.voice_similarity(audio, enrolled)
        how = f"Both models ran live on this machine in {time.time() - t0:.1f} s."
    name = "you" if who == ME else accounts[who]["name"]
    sig = {"synthetic_voice": synthetic_voice.analyze(p, secs),
           "voiceprint": voiceprint.analyze(sim, secs, {"name": name})}
    d = fusion.fuse(sig)
    _result(d, p, sim, how)


def _result(d, p: float, sim: float, how: str) -> None:
    color, icon, label = S.ACTION[d.action]
    st.markdown(f'<div class="vs-card"><div class="vs-kicker">Verdict from the two voice models</div>'
                f'<div class="vs-verdict" style="color:{color};font-size:1.5rem">{S.esc(d.verdict)}</div>'
                f'<div style="margin:4px 0 8px">{S.badge(f"{icon} {label} · risk {d.risk:.0f}%", color)}</div>'
                + "".join(f'<div class="vs-reason">• {S.esc(r)}</div>' for r in d.reasons) + "</div>",
                unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Synthetic voice (Spectra-AASIST)", f"{p:.0%}", help="Above 50% = the model card's spoof threshold.")
    lo, hi = voiceprint.THRESHOLDS
    word = "match" if sim >= hi else "no match" if sim <= lo else "weak match"
    m2.metric("Voiceprint similarity (NVIDIA TitaNet)", f"{sim:.2f}", help=f"Match at {hi} or above, mismatch at {lo} or below.")
    m2.caption(word)
    st.caption(how + " In a real call these two scores are fused with conversation behaviour, caller ID, "
               "account context and the liveness challenge (see the Call co-pilot).")
