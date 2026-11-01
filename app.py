"""Deepfake Voice Shield - agent co-pilot dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import html
import time

import plotly.graph_objects as go
import streamlit as st

from vshield import handoff
from vshield.session import DATA, CallSession, list_calls, load_accounts, load_call

st.set_page_config(page_title="Deepfake Voice Shield", page_icon="🛡️", layout="wide")

GREEN, AMBER, RED, INK, MUTED = "#1a7f4b", "#c98a00", "#cd040b", "#111111", "#6b6b6b"
SIGNAL_ORDER = ["acoustic", "voiceprint", "behavior", "challenge", "telecom", "account"]


def color(risk: float) -> str:
    return GREEN if risk < 40 else AMBER if risk < 75 else RED


st.markdown(f"""
<style>
  .block-container {{padding-top: 3.2rem; max-width: 1400px;}}
  .brand {{font-size: 1.7rem; line-height: 1.2; font-weight: 800; letter-spacing: -0.02em; margin: 0;}}
  .brand span {{color: {RED};}}
  .sub {{color: {MUTED}; margin-top: -2px; margin-bottom: 0.6rem;}}
  .bubble {{padding: 10px 14px; border-radius: 12px; margin: 6px 0; max-width: 88%; line-height: 1.4; font-size: 0.95rem;}}
  .agent  {{background: rgba(127,127,127,0.10); margin-right: auto;}}
  .caller {{background: rgba(205,4,11,0.06); border: 1px solid rgba(205,4,11,0.18); margin-left: auto;}}
  .who {{font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED}; margin-bottom: 2px;}}
  .tag {{display:inline-block; font-size:0.7rem; padding:1px 7px; border-radius:999px; margin-left:6px;
        border:1px solid currentColor;}}
  .verdict {{font-size: 1.25rem; font-weight: 750; margin: 0 0 4px 0;}}
  .action {{border-left: 4px solid; padding: 8px 12px; border-radius: 4px; background: rgba(127,127,127,0.07);
           font-size: 0.92rem; margin-bottom: 10px;}}
  .sig {{margin: 9px 0;}}
  .sig-top {{display:flex; justify-content:space-between; font-size:0.88rem; font-weight:600;}}
  .track {{height: 8px; background: rgba(127,127,127,0.15); border-radius: 99px; overflow: hidden; margin: 4px 0 3px;}}
  .fill {{height: 100%; border-radius: 99px;}}
  .why {{font-size: 0.78rem; color: {MUTED}; line-height: 1.3;}}
  .challenge {{border: 1.5px dashed {AMBER}; border-radius: 10px; padding: 10px 14px; margin: 8px 0 12px;}}
  .reason {{font-size: 0.88rem; margin: 3px 0;}}
</style>
""", unsafe_allow_html=True)

# ---------------- state ----------------
STORY = ["genuine_customer", "voice_clone", "ai_negotiator"]
calls = {cid: load_call(cid) for cid in sorted(list_calls(), key=lambda c: STORY.index(c) if c in STORY else 99)}
accounts = load_accounts()

with st.sidebar:
    st.markdown("### Demo call")
    cid = st.radio("Incoming call", list(calls), format_func=lambda c: calls[c]["title"], label_visibility="collapsed")
    if st.session_state.get("cid") != cid:
        st.session_state.cid = cid
        st.session_state.sess = CallSession(calls[cid], accounts)
        st.session_state.sess.start()
        st.session_state.playing = False
    sess: CallSession = st.session_state.sess

    c1, c2, c3 = st.columns(3)
    if c1.button("Next ▸", width="stretch", disabled=sess.done):
        sess.add_turn()
    if c2.button("Play ⏵", width="stretch", disabled=sess.done):
        st.session_state.playing = True
    if c3.button("Reset", width="stretch"):
        st.session_state.sess = sess = CallSession(calls[cid], accounts)
        sess.start()
        st.session_state.playing = False
    speed = st.slider("Playback delay (s per turn)", 0.3, 3.0, 1.2, 0.1)

    st.markdown("### Caller audio")
    st.audio(str(DATA / calls[cid]["caller_audio"]))
    st.caption("Synthetic audio generated for the demo - no real customer data.")
    if st.toggle("Reveal ground truth"):
        st.info(f"Ground truth: **{calls[cid]['ground_truth']}**")
    st.markdown("---")
    st.caption("Backends: demo DSP models by default. Set `VSHIELD_VOICEPRINT=titanet` and "
               "`VSHIELD_SPOOF_MODEL=hf:<model>` to use NVIDIA TitaNet and a trained anti-spoofing model.")

call, acc = sess.call, sess.account
snap = sess.history[-1]
risk_c = color(snap.risk)

st.markdown('<div class="brand">Deepfake Voice <span>Shield</span></div>', unsafe_allow_html=True)
t = call["telecom"]
st.markdown(f'<p class="sub">Agent co-pilot &nbsp;·&nbsp; {call["account_id"]} {html.escape(acc["name"])} &nbsp;·&nbsp; '
            f'ANI {t["ani"]} &nbsp;·&nbsp; IVR intent: {t["ivr_intent"].replace("_", " ")}</p>', unsafe_allow_html=True)

left, right = st.columns([1.35, 1], gap="large")

# ---------------- transcript ----------------
with left:
    st.markdown("#### Live transcript")
    box = st.container(height=470)
    with box:
        st.caption("Newest turn at the top")
        if not sess.turns:
            st.caption("Call ringing - telecom and account context already scored (pre-answer risk).")
        for tr in reversed(sess.turns):
            tags = ""
            if tr.challenge:
                tags += f'<span class="tag" style="color:{AMBER}">liveness challenge</span>'
            if tr.interrupts:
                tags += f'<span class="tag" style="color:{MUTED}">barge-in test</span>'
            who = "Agent" if tr.speaker == "agent" else "Caller"
            st.markdown(f'<div class="bubble {tr.speaker}"><div class="who">{who} · {tr.start:.1f}s{tags}</div>'
                        f'{html.escape(tr.text)}</div>', unsafe_allow_html=True)

    st.markdown("#### Risk over the call")
    xs = ["ring"] + [f"T{i}" for i in range(1, len(sess.history))]
    fig = go.Figure(go.Scatter(x=xs, y=[h.risk for h in sess.history], mode="lines+markers",
                               line=dict(color=risk_c, width=3), marker=dict(size=7)))
    fig.add_hrect(y0=0, y1=40, fillcolor=GREEN, opacity=0.07, line_width=0)
    fig.add_hrect(y0=40, y1=75, fillcolor=AMBER, opacity=0.07, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor=RED, opacity=0.07, line_width=0)
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 100], title="risk %"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---------------- co-pilot panel ----------------
with right:
    g = go.Figure(go.Indicator(
        mode="gauge+number", value=snap.risk, number={"suffix": "%", "valueformat": ".0f", "font": {"size": 44, "color": risk_c}},
        title={"text": "Bot / fraud likelihood", "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": risk_c, "thickness": 0.3},
               "steps": [{"range": [0, 40], "color": "rgba(26,127,75,0.15)"},
                         {"range": [40, 75], "color": "rgba(201,138,0,0.15)"},
                         {"range": [75, 100], "color": "rgba(205,4,11,0.15)"}]}))
    g.update_layout(height=215, margin=dict(l=35, r=45, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(g, width="stretch", config={"displayModeBar": False})

    st.markdown(f'<p class="verdict" style="color:{risk_c}">{html.escape(snap.verdict)}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="action" style="border-color:{risk_c}"><b>{snap.action.replace("_", " ")}</b> — '
                f'{html.escape(snap.action_text)}</div>', unsafe_allow_html=True)

    if sess.turns and sess.turns[-1].challenge:
        st.markdown('<div class="challenge"><b>Challenge issued</b> - waiting for the caller. Shield is timing the response.</div>',
                    unsafe_allow_html=True)
    if snap.suggested_challenge:
        sc = snap.suggested_challenge
        st.markdown(f'<div class="challenge"><b>Suggested liveness challenge</b><br>{html.escape(sc["agent_script"])}'
                    f'<div class="why" style="margin-top:4px">{html.escape(sc["why"])}</div></div>',
                    unsafe_allow_html=True)
    for r in snap.challenge_results:
        c = GREEN if r["passed"] else RED
        st.markdown(f'<div class="action" style="border-color:{c}"><b>Challenge "{html.escape(r["phrase"])}"</b>: '
                    f'{html.escape(r["verdict"])}</div>', unsafe_allow_html=True)

    if snap.top_reasons:
        st.markdown("**Why**")
        for r in snap.top_reasons:
            st.markdown(f'<div class="reason">• {html.escape(r)}</div>', unsafe_allow_html=True)

    st.markdown("**Signals** <span class='why'>(each bar = how strongly that signal points to fraud)</span>",
                unsafe_allow_html=True)
    for k in SIGNAL_ORDER:
        s = snap.signals.get(k)
        if not s:
            continue
        pct = s.score * 100
        c = color(pct) if s.available else MUTED
        val = f"{pct:.0f}%" if s.available else "waiting"
        st.markdown(f'<div class="sig"><div class="sig-top"><span>{s.label}</span><span style="color:{c}">{val}</span></div>'
                    f'<div class="track"><div class="fill" style="width:{pct if s.available else 0:.0f}%;background:{c}"></div></div>'
                    f'<div class="why">{html.escape(s.reasons[0])}</div></div>', unsafe_allow_html=True)

# ---------------- generated documents ----------------
st.markdown("---")
st.markdown("#### Generated documents")
card = handoff.handoff_card(call, acc, snap, sess.history)
tabs = st.tabs(["Transfer handoff card", "Fraud referral", "Raw signal data"])
with tabs[0]:
    st.markdown(card)
    st.download_button("Download handoff card", card, file_name=f"handoff_{call['call_id']}.md")
with tabs[1]:
    if snap.action in ("ESCALATE", "STEP_UP"):
        ref = handoff.fraud_referral(call, acc, snap)
        st.markdown(ref)
        st.download_button("Download fraud referral", ref, file_name=f"fraud_referral_{call['call_id']}.md")
    else:
        st.caption("No referral - caller assessed as genuine.")
with tabs[2]:
    st.json({k: {"score": round(v.score, 3), "weight": round(v.weight, 2), "available": v.available,
                 "reasons": v.reasons, "details": v.details} for k, v in snap.signals.items()}, expanded=False)

# ---------------- autoplay ----------------
if st.session_state.get("playing"):
    if sess.done:
        st.session_state.playing = False
    else:
        time.sleep(speed)
        sess.add_turn()
        st.rerun()
