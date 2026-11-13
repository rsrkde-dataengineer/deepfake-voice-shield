"""Shared look: colors, CSS and small HTML building blocks used by every page."""
from __future__ import annotations

import html

import streamlit as st

# Status colors (fixed meaning, always shown with an icon + a word) and chart series colors (validated set).
GOOD, WARNING, CRITICAL, INFO = "#0ca30c", "#fab219", "#d03b3b", "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua - in this order, never cycled
MUTED = "#898781"
BRAND = "#cd040b"

ACTION = {   # action -> (color, icon, short label)
    "PROCEED": (GOOD, "✅", "Proceed"),
    "STEP_UP": (WARNING, "⚠️", "Step up"),
    "ESCALATE": (CRITICAL, "⛔", "Escalate to Fraud"),
    "AUTHORIZED_AGENT": (INFO, "🤖", "Authorised AI agent"),
}
CALL_ICONS = {"genuine_customer": ":material/person:", "genuine_new_number": ":material/flight:",
              "voice_clone": ":material/theater_comedy:", "human_impersonator": ":material/no_accounts:",
              "ai_negotiator": ":material/smart_toy:", "human_negotiator": ":material/handshake:",
              "ai_assistant": ":material/support_agent:"}


def level(score: float) -> tuple[str, str]:
    """0..1 fraud likelihood -> (status color, word)."""
    return (GOOD, "low") if score < 0.4 else (WARNING, "medium") if score < 0.8 else (CRITICAL, "high")


def esc(text: str) -> str:
    return html.escape(str(text))


CSS = f"""
<style>
  .block-container {{padding-top: 3.6rem; max-width: 1440px;}}
  .vs-title {{font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; line-height: 1.2;}}
  .vs-title span {{color: {BRAND};}}
  .vs-sub {{opacity: 0.72; margin: 2px 0 14px 0; font-size: 0.95rem;}}
  .vs-card {{border: 1px solid rgba(127,127,127,0.25); border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;}}
  .vs-kicker {{font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.65; margin-bottom: 4px;}}
  .vs-hero {{font-size: 3.4rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em;}}
  .vs-verdict {{font-size: 1.2rem; font-weight: 750; margin: 6px 0 2px 0;}}
  .vs-badge {{display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700;}}
  .vs-meter {{position: relative; height: 10px; border-radius: 99px; background: rgba(127,127,127,0.16); margin: 12px 0 4px;}}
  .vs-meter .fill {{position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px;}}
  .vs-meter .tick {{position: absolute; top: -3px; bottom: -3px; width: 2px; background: rgba(127,127,127,0.55);}}
  .vs-meter-scale {{display: flex; justify-content: space-between; font-size: 0.72rem; opacity: 0.65;}}
  .vs-action {{border-left: 4px solid; padding: 9px 12px; border-radius: 6px; background: rgba(127,127,127,0.07);
              font-size: 0.92rem; margin: 10px 0;}}
  .vs-bubble {{padding: 9px 13px; border-radius: 12px; margin: 6px 0; max-width: 90%; line-height: 1.42; font-size: 0.94rem;}}
  .vs-agent {{background: rgba(127,127,127,0.10); margin-right: auto;}}
  .vs-caller {{background: rgba(205,4,11,0.06); border: 1px solid rgba(205,4,11,0.18); margin-left: auto;}}
  .vs-who {{font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.65; margin-bottom: 2px;}}
  .vs-tag {{display: inline-block; font-size: 0.68rem; padding: 0 7px; border-radius: 999px; margin-left: 6px;
           border: 1px solid currentColor; text-transform: none; letter-spacing: 0;}}
  .vs-latest {{box-shadow: 0 0 0 2px rgba(205,4,11,0.35);}}
  .vs-sig {{margin: 10px 0;}}
  .vs-sig-top {{display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 650;}}
  .vs-track {{height: 7px; background: rgba(127,127,127,0.16); border-radius: 99px; overflow: hidden; margin: 4px 0 3px;}}
  .vs-track div {{height: 100%; border-radius: 99px;}}
  .vs-why {{font-size: 0.8rem; opacity: 0.75; line-height: 1.35;}}
  .vs-challenge {{border: 1.5px dashed {WARNING}; border-radius: 10px; padding: 10px 14px; margin: 10px 0;}}
  .vs-reason {{font-size: 0.9rem; margin: 3px 0;}}
  .vs-small {{font-size: 0.8rem; opacity: 0.7;}}
  [data-testid="stMetricLabel"] * {{white-space: normal !important; overflow: visible !important;
                                    text-overflow: clip !important;}}   /* wrap long KPI labels, don't cut them */
</style>
"""


def top_nav(pages: list) -> None:
    """Website-style links at the top right of every page (About, Vision, Value)."""
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        for page in pages:
            st.page_link(page)


def header(title_html: str, sub: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<p class="vs-title">{title_html}</p>' + (f'<p class="vs-sub">{sub}</p>' if sub else ""),
                unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    ink = "#0b0b0b" if color == WARNING else "#ffffff"
    return f'<span class="vs-badge" style="background:{color};color:{ink}">{esc(text)}</span>'


def meter(risk: float, color: str) -> str:
    return (f'<div class="vs-meter"><div class="fill" style="width:{risk:.0f}%;background:{color}"></div>'
            f'<div class="tick" style="left:40%"></div><div class="tick" style="left:80%"></div></div>'
            '<div class="vs-meter-scale"><span>0 · proceed</span><span style="margin-left:22%">40 · step up</span>'
            '<span>80 · escalate · 100</span></div>')


def signal_row(label: str, score: float, available: bool, reason: str) -> str:
    if not available:
        return (f'<div class="vs-sig"><div class="vs-sig-top"><span>{esc(label)}</span><span style="opacity:.6">waiting</span></div>'
                f'<div class="vs-track"><div style="width:0"></div></div><div class="vs-why">{esc(reason)}</div></div>')
    color, word = level(score)
    return (f'<div class="vs-sig"><div class="vs-sig-top"><span>{esc(label)}</span><span>{score:.0%} · {word}</span></div>'
            f'<div class="vs-track"><div style="width:{score * 100:.0f}%;background:{color}"></div></div>'
            f'<div class="vs-why">{esc(reason)}</div></div>')


def plotly_layout(fig, height: int, **kw) -> None:
    """Quiet chrome: transparent background, hairline grid. Keyword arguments override the defaults."""
    layout = dict(height=height, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified")
    fig.update_layout(**(layout | kw))
    fig.update_xaxes(showgrid=False, linecolor="rgba(127,127,127,0.4)")
    fig.update_yaxes(gridcolor="rgba(127,127,127,0.18)", zeroline=False)
