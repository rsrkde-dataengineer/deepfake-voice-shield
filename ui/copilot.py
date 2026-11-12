"""Page 1 - Agent co-pilot: replay a call turn by turn and watch Shield decide in real time."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import style as S
from vshield import detectors, handoff
from vshield.audio import telephone, wav_bytes
from vshield.session import CallSession, call_audio, list_calls, load_accounts, load_call

SIGNAL_ORDER = ["synthetic_voice", "voiceprint", "behavior", "challenge", "telecom", "account"]


@st.cache_data
def _calls() -> dict:
    return {cid: load_call(cid) for cid in list_calls()}


@st.cache_data
def _accounts() -> dict:
    return load_accounts()


@st.cache_data
def _audio(cid: str, phone: bool) -> tuple[np.ndarray, int]:
    y, sr = call_audio(_calls()[cid])
    return (telephone(y, sr) if phone else y), sr


# ---- session state -------------------------------------------------------------------------------------
def _start(cid: str, live: bool) -> None:
    sess = CallSession(_calls()[cid], _accounts(), live=live)
    with st.spinner("Loading Spectra-AASIST and NVIDIA TitaNet...") if live else st.empty():
        sess.start()
    st.session_state.update(sess=sess, cid=cid, live=live, playing=False, just_added=None)


def _advance() -> None:
    sess = st.session_state.sess
    if not sess.done:
        sess.add_turn()
        st.session_state.just_added = len(sess.turns) - 1


def _play() -> None:
    st.session_state.playing = True
    _advance()


def _pause() -> None:
    st.session_state.playing = False


def _reset() -> None:
    _start(st.session_state.cid, st.session_state.live)


# ---- page --------------------------------------------------------------------------------------------
def render() -> None:
    calls = _calls()
    with st.sidebar:
        st.markdown("#### Incoming call")
        cid = st.radio("Call", list(calls), label_visibility="collapsed",
                       format_func=lambda c: f"{S.CALL_ICONS.get(c, ':material/call:')} {calls[c]['title']}")
        can_live = detectors.available()
        live = st.toggle("Run the models live", value=False, disabled=not can_live,
                         help="On: Spectra-AASIST and NVIDIA TitaNet run on this machine. Off: the scores those same "
                              "models produced ahead of time." if can_live else
                              "Install requirements-ml.txt to run the models live on this machine.")
        if st.session_state.get("cid") != cid or st.session_state.get("live") != live:
            _start(cid, live)
        sess: CallSession = st.session_state.sess
        playing = st.session_state.playing
        b1, b2, b3 = st.columns([1, 1, 0.5])
        b1.button("Next", on_click=_advance, disabled=sess.done or playing, width="stretch", type="primary",
                  help="Play the next turn of the call")
        if playing:
            b2.button("Pause", on_click=_pause, width="stretch")
        else:
            b2.button("Play", on_click=_play, disabled=sess.done, width="stretch", help="Play the rest of the call")
        b3.button(":material/replay:", on_click=_reset, width="stretch", help="Restart this call")
        hear = st.toggle("Hear each turn", value=True, help="Plays every turn's audio as it happens.")
        phone = st.toggle("Phone-line sound (8 kHz)", value=False, help="Hear the call as a narrowband G.711 phone line.")
        st.markdown("#### Whole call recording")
        y, sr = _audio(cid, phone)
        st.audio(wav_bytes(y, sr))
        call = calls[cid]
        st.caption(f"Caller: {call['voices']['caller']}. Agent: {call['voices']['agent']}.")
        if st.toggle("Reveal ground truth"):
            st.info(f"Ground truth: **{call['ground_truth']}** - expected action **{call['expected_action']}**")

    acc = sess.account
    t = call["telecom"]
    S.header('Deepfake Voice <span>Shield</span> · agent co-pilot',
             f"{S.esc(acc['name'])} ({call['account_id']}) · caller ID {t['ani']} · STIR/SHAKEN {t['stir_shaken']} · "
             f"{t['line_type']} · reason: {t['ivr_intent'].replace('_', ' ')} · agent {call['agent_name']}")
    st.caption(call["summary"])
    snap = sess.history[-1]

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        _transcript(sess, y, sr, hear)
        _risk_chart(sess)
    with right:
        _decision(sess, snap)
    st.divider()
    _documents(sess, snap, call)

    if st.session_state.playing:      # auto-advance, pacing each turn by its own length
        if sess.done:
            st.session_state.playing = False
            st.rerun()
        last = sess.turns[-1] if sess.turns else None
        time.sleep(max(0.8, (last.end - last.start) + 0.4) if (last and hear) else 1.2)
        _advance()
        st.rerun()


def _transcript(sess: CallSession, y: np.ndarray, sr: int, hear: bool) -> None:
    st.markdown("##### Live transcript")
    if hear and st.session_state.just_added is not None and sess.turns:
        turn = sess.turns[-1]
        seg = y[int(turn.start * sr): int((turn.end + 0.15) * sr)]
        st.audio(wav_bytes(seg, sr), autoplay=True)
        st.session_state.just_added = None
    box = st.container(height=440)
    with box:
        if not sess.turns:
            st.caption("Call ringing. Caller ID and account context are already scored before anyone speaks.")
        for i, tr in enumerate(reversed(sess.turns)):
            tags = ""
            if tr.challenge:
                tags += f'<span class="vs-tag" style="color:{S.WARNING}">liveness challenge</span>'
            if tr.interrupts:
                tags += '<span class="vs-tag">barge-in test</span>'
            if tr.event == "consent_approved":
                tags += f'<span class="vs-tag" style="color:{S.GOOD}">customer approved on phone</span>'
            who = f"Agent {sess.call['agent_name']}" if tr.speaker == "agent" else "Caller"
            latest = " vs-latest" if i == 0 else ""
            st.markdown(f'<div class="vs-bubble vs-{tr.speaker}{latest}"><div class="vs-who">{who} · {tr.start:.1f}s'
                        f'{tags}</div>{S.esc(tr.text)}</div>', unsafe_allow_html=True)


def _risk_chart(sess: CallSession) -> None:
    st.markdown("##### Risk over the call")
    hist = sess.history
    xs = ["ring"] + [f"T{i}" for i in range(1, len(hist))]
    texts = ["Before answer: caller ID + account"] + [
        f"{sess.turns_all[h.turn_idx].speaker}: {sess.turns_all[h.turn_idx].text[:60]}" for h in hist[1:]]
    colors = [S.ACTION[h.action][0] for h in hist]
    fig = go.Figure()
    for lo, hi, color, name in ((0, 40, S.GOOD, "proceed"), (40, 80, S.WARNING, "step up"), (80, 100, S.CRITICAL, "escalate")):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.07, line_width=0,
                      annotation_text=name, annotation_position="top right", annotation_font_size=11,
                      annotation_font_color=S.MUTED)
    fig.add_trace(go.Scatter(x=xs, y=[h.risk for h in hist], mode="lines+markers", line=dict(color=S.MUTED, width=2),
                             marker=dict(size=10, color=colors, line=dict(width=2, color="rgba(255,255,255,0.9)")),
                             customdata=[[S.ACTION[h.action][2], tx] for h, tx in zip(hist, texts)],
                             hovertemplate="%{y:.0f}% · %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    S.plotly_layout(fig, 230, showlegend=False)
    fig.update_yaxes(range=[0, 100], title=None, ticksuffix="%")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, config={"displayModeBar": False})


def _decision(sess: CallSession, snap) -> None:
    color, icon, label = S.ACTION[snap.action]
    agent_call = snap.action == "AUTHORIZED_AGENT"
    kicker = "Bot likelihood - disclosed AI assistant, not fraud" if agent_call else "Fraud / bot likelihood"
    if snap.action != "AUTHORIZED_AGENT":
        color = S.GOOD if snap.risk < 40 else S.WARNING if snap.risk < 80 else S.CRITICAL
    st.markdown(f'<div class="vs-card"><div class="vs-kicker">{kicker}</div>'
                f'<div class="vs-hero" style="color:{color}">{snap.risk:.0f}%</div>'
                f'<div class="vs-verdict">{S.esc(snap.verdict)}</div>{S.badge(f"{icon} {label}", S.ACTION[snap.action][0])}'
                f'{S.meter(snap.risk, color)}'
                f'<div class="vs-action" style="border-color:{S.ACTION[snap.action][0]}">{S.esc(snap.action_text)}</div></div>',
                unsafe_allow_html=True)

    if sess.turns and sess.turns[-1].challenge:
        st.markdown('<div class="vs-challenge"><b>Challenge asked</b> - waiting for the caller. '
                    'Shield is timing how long the answer takes.</div>', unsafe_allow_html=True)
    if snap.suggested_challenge:
        sc = snap.suggested_challenge
        st.markdown(f'<div class="vs-challenge"><b>Suggested liveness challenge</b><br>{S.esc(sc["agent_script"])}'
                    f'<div class="vs-why" style="margin-top:4px">{S.esc(sc["why"])}</div></div>', unsafe_allow_html=True)
    for r in snap.challenge_results:
        c = S.GOOD if r["passed"] else S.CRITICAL
        st.markdown(f'<div class="vs-action" style="border-color:{c}">{"✅" if r["passed"] else "⛔"} '
                    f'<b>Challenge "{S.esc(r["phrase"])}"</b>: {S.esc(r["verdict"])}</div>', unsafe_allow_html=True)

    if snap.top_reasons:
        st.markdown("**Why**")
        st.markdown("".join(f'<div class="vs-reason">• {S.esc(r)}</div>' for r in snap.top_reasons), unsafe_allow_html=True)

    st.markdown("**Signals** <span class='vs-small'>(how strongly each one points to fraud)</span>", unsafe_allow_html=True)
    rows = [S.signal_row(snap.signals[k].label, snap.signals[k].score, snap.signals[k].available, snap.signals[k].reasons[0])
            for k in SIGNAL_ORDER if k in snap.signals]
    st.markdown("".join(rows), unsafe_allow_html=True)
    mode = "running live on this machine" if sess.live else "scores produced by these models ahead of time"
    st.caption(f"Synthetic voice: {detectors.MODEL_NAMES['synthetic_voice']}. Voiceprint: "
               f"{detectors.MODEL_NAMES['voiceprint']}. Model scores: {mode}.")


def _documents(sess: CallSession, snap, call: dict) -> None:
    st.markdown("#### Generated for the next agent")
    card = handoff.handoff_card(call, sess.account, snap, sess.history)
    tabs = st.tabs(["Transfer handoff card", "Fraud referral", "Model evidence", "Raw signals"])
    with tabs[0]:
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            st.markdown(card)
        with c2:
            verified = handoff.is_verified(snap)
            st.metric("Re-verification after transfer", "Skipped" if verified else "Required",
                      help="A verified caller's handoff card replaces the knowledge questions on transfer.")
            st.metric("Agent time saved on this transfer", "60-90 s" if verified else "0 s")
            st.download_button("Download handoff card", card, file_name=f"handoff_{call['call_id']}.md",
                               icon=":material/download:")
    with tabs[1]:
        if snap.action in ("STEP_UP", "ESCALATE"):
            ref = handoff.fraud_referral(call, sess.account, snap)
            st.markdown(ref)
            st.download_button("Download fraud referral", ref, file_name=f"fraud_referral_{call['call_id']}.md",
                               icon=":material/download:")
        else:
            st.caption("No referral: this call is not assessed as fraud.")
    with tabs[2]:
        rows = []
        for tr in sess.turns:
            if tr.speaker == "caller" and tr.scores:
                rows.append({"Turn": f"T{tr.idx + 1}", "Caller speech so far (s)": tr.audio_span[1],
                             "Synthetic voice (Spectra-AASIST)": tr.scores["synthetic_voice"],
                             "Voiceprint similarity (TitaNet)": tr.scores["voice_similarity"]})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Synthetic voice (Spectra-AASIST)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Voiceprint similarity (TitaNet)": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Each model scores all caller audio heard so far. Synthetic above 50% = the model card's spoof "
                       "threshold. Voiceprint: match at 0.70 or above, mismatch at 0.55 or below (calibrated on "
                       "LibriSpeech dev-clean, checked on test-clean - see Model evaluation).")
        else:
            st.caption("No caller speech yet.")
    with tabs[3]:
        st.json({k: {"score": round(v.score, 3), "weight": round(v.weight, 2), "available": v.available,
                     "reasons": v.reasons, "details": v.details} for k, v in snap.signals.items()}, expanded=False)
