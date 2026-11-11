"""Page 3 - Model evaluation: how well the two models separate real voices from fakes (reports/eval.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import style as S

REPORT = Path(__file__).resolve().parents[1] / "reports" / "eval.json"
SPLITS = {"test-clean": "Held-out test (test-clean)", "dev-clean": "Calibration (dev-clean)"}
CHANNELS = {"telephone": "Phone line (8 kHz G.711)", "clean": "Clean 16 kHz audio"}
ATTACK_NAMES = {"chatterbox_clone": "Voice clones (Chatterbox)", "kokoro": "Neural TTS (Kokoro)", "macos_say": "Older TTS (macOS)"}
REAL, CLONE, TTS = S.SERIES   # each kind of voice keeps one color on every chart


def _hist(fig: go.Figure, values: list[float], name: str, color: str, bins: np.ndarray) -> None:
    counts, edges = np.histogram(values, bins=bins)
    share = counts / max(len(values), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Scatter(x=centers, y=share, name=f"{name} ({len(values)})", mode="lines",
                             line=dict(color=color, width=2, shape="hvh"),
                             hovertemplate=f"{name}<br>score %{{x:.2f}}: %{{y:.0%}} of clips<extra></extra>"))


def render() -> None:
    S.header("Model <span>evaluation</span>",
             "How well Spectra-AASIST and NVIDIA TitaNet separate real voices from clones and TTS - measured, not claimed.")
    if not REPORT.exists():
        st.info("No evaluation report yet. Build the sets with `scripts/build_eval_set.py`, then run "
                "`scripts/evaluate.py` (see README).")
        return
    rep = json.loads(REPORT.read_text())
    c1, c2 = st.columns(2)
    split = c1.segmented_control("Data", [s for s in SPLITS if s in rep["splits"]], format_func=SPLITS.get,
                                 default=next(s for s in SPLITS if s in rep["splits"]), required=True)
    ch = c2.segmented_control("Channel", list(CHANNELS), format_func=CHANNELS.get, default="telephone", required=True)
    res = rep["splits"][split]
    m = res["channels"][ch]
    sv, vp = m["synthetic_voice"], m["voiceprint"]
    n = res["counts"]

    k = st.columns(5)
    caught = np.mean(list(sv["detection_rate"].values()))
    k[0].metric("Fakes caught", f"{caught:.0%}", help="Fake voices flagged as synthetic (clones, neural TTS and older "
                                                      "TTS), at the model card's 0.5 threshold.", border=True)
    k[1].metric("Real voices flagged", f"{sv['false_alarm_rate']:.1%}",
                help=f"Real human test clips wrongly flagged as synthetic, out of {n['real_test_clips']}. With zero errors "
                     f"in {n['real_test_clips']}, the true rate is below ~{3 / n['real_test_clips']:.1%} with 95% "
                     f"confidence (rule of three).", border=True)
    k[2].metric("Voiceprint fooled", f"{vp['accept_rate']['clone']:.0%}",
                help="Share of voice clones that score at or above the 0.70 match threshold: what a voiceprint-only "
                     "system would let through.", border=True)
    fooled, flagged = m["clones_fooling_voiceprint"], m["of_those_flagged_synthetic"]
    k[3].metric("Shield catches", f"{flagged} of {fooled}" if fooled else "-",
                help="Of the clones that fooled the voiceprint, how many the anti-spoofing model flagged.", border=True)
    k[4].metric("Voiceprint EER", f"{vp['eer']:.1%}",
                help="Equal error rate between same-speaker and different-speaker trials of real voices.", border=True)
    st.caption(f"{n['speakers']} real speakers, {n['real_test_clips']} real test clips and {n['fake_test_clips']} fake ones "
               f"({', '.join(f'{v} {ATTACK_NAMES[a[:-6]][0].lower()}{ATTACK_NAMES[a[:-6]][1:]}' for a, v in n.items() if a.endswith('_clips') and a[:-6] in ATTACK_NAMES)}).")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Voiceprint: who is speaking? (NVIDIA TitaNet)")
        fig = go.Figure()
        bins = np.linspace(-0.3, 1.0, 40)
        for key, name, color in (("other_speaker", "Different speaker", S.MUTED), ("same_speaker", "Same speaker", REAL),
                                 ("clone", "Voice clone", CLONE)):
            _hist(fig, vp["scores"][key], name, color, bins)
        fig.add_vline(x=rep["decision_points"]["voiceprint_match"], line=dict(color=S.MUTED, width=1.5),
                      annotation_text="match ≥ 0.70", annotation_position="top left", annotation_font_size=11)
        S.plotly_layout(fig, 340, hovermode="closest", margin=dict(l=8, r=8, t=40, b=8))
        fig.update_xaxes(title="cosine similarity to the enrolled voice")
        fig.update_yaxes(title="share of trials", tickformat=".0%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Real speakers separate cleanly, but voice clones land among the genuine matches. "
                   "A voiceprint alone cannot stop them.")
    with right:
        st.markdown("##### Synthetic voice: is it a real human? (Spectra-AASIST)")
        fig = go.Figure()
        bins = np.linspace(0, 1, 26)
        tts = sv["scores"].get("kokoro", []) + sv["scores"].get("macos_say", [])
        for vals, name, color in ((sv["scores"]["real"], "Real human", REAL),
                                  (sv["scores"].get("chatterbox_clone", []), "Voice clone", CLONE),
                                  (tts, "Text-to-speech", TTS)):
            _hist(fig, vals, name, color, bins)
        fig.add_vline(x=rep["decision_points"]["synthetic_voice"], line=dict(color=S.MUTED, width=1.5),
                      annotation_text="spoof ≥ 0.5", annotation_position="top right", annotation_font_size=11)
        S.plotly_layout(fig, 340, hovermode="closest", margin=dict(l=8, r=8, t=40, b=8))
        fig.update_xaxes(title="synthetic-speech probability")
        fig.update_yaxes(title="share of clips", tickformat=".0%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("The anti-spoofing model puts the clones and TTS voices on the other side of the threshold - "
                   "including the clones that fooled the voiceprint.")

    st.markdown("##### All results")
    rows = []
    for s, r in rep["splits"].items():
        for c, mm in r["channels"].items():
            row = {"Data": SPLITS[s], "Channel": CHANNELS[c],
                   "Anti-spoofing EER": mm["synthetic_voice"]["eer"],
                   "Real callers flagged": mm["synthetic_voice"]["false_alarm_rate"]}
            row |= {f"Caught: {ATTACK_NAMES[a].split(' (')[0]}": v for a, v in mm["synthetic_voice"]["detection_rate"].items()}
            row |= {"Voiceprint EER": mm["voiceprint"]["eer"], "Clones accepted by voiceprint": mm["voiceprint"]["accept_rate"]["clone"],
                    "Different speakers accepted": mm["voiceprint"]["accept_rate"]["other_speaker"]}
            rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in df.columns[2:]})

    with st.expander("Method and limits"):
        st.markdown(f"""
- **Real voices:** LibriSpeech (CC BY 4.0). Thresholds were set on **dev-clean**; **test-clean** is held out:
  new speakers, new clone targets and TTS voices used nowhere else. The TitaNet protocol (same-speaker vs
  different-speaker trials on test-clean) follows the team's verification notebook.
- **Fake voices:** Chatterbox zero-shot clones made from a separate recording of each target speaker, Kokoro-82M
  neural TTS, and macOS text-to-speech.
- **Decision points fixed in advance:** synthetic probability 0.5 (Spectra-AASIST model card) and voiceprint match
  0.70 (dev-clean only). Equal error rates are shown for comparison with published results.
- **Phone line:** every clip also goes through 8 kHz G.711; mobile codecs (AMR, EVS, Opus) and packet loss are not
  covered yet.
- **Limits:** small sets ({n['real_test_clips']} real clips per split can't measure a 1% false-alarm rate precisely);
  read audiobook speech, not phone conversations; TitaNet's training data includes LibriSpeech (other speakers), which
  flatters its speaker separation; newer or adversarially tuned generators may evade the detector (Spectra-AASIST
  reports 14% EER on the harder ASVspoof5 benchmark). Generated on {rep['generated']}.
""")
