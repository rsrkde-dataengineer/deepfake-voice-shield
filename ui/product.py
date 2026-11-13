"""Product pages - About, Vision and Value: the story around the demo (text in ui/content.py, money in vshield.impact)."""
from __future__ import annotations

import streamlit as st

from ui import content as C
from ui import style as S
from vshield import impact

LINKS: dict = {}   # pages to link to (copilot, evaluation, impact, how), filled in by app.py

CSS = """
<style>
  .vs-band {background: #141414; border-radius: 14px; padding: 26px 30px 24px; margin: 6px 0 22px;}
  .vs-band .k {font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #a8a6a0; margin-bottom: 8px;}
  .vs-band .t {font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.12; color: #ffffff;
               margin-bottom: 10px;}
  .vs-band .t span {color: #ff4d4d;}
  .vs-band .p {font-size: 1.02rem; line-height: 1.5; color: #d6d4ce; max-width: 980px;}
  .vs-band .stats {display: flex; flex-wrap: wrap; gap: 14px 38px; margin-top: 18px;}
  .vs-band .stats b {display: block; font-size: 1.6rem; color: #ffffff; letter-spacing: -0.02em;}
  .vs-band .stats span {font-size: 0.82rem; color: #a8a6a0;}
  .vs-step {display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 50%;
            background: #cd040b; color: #fff; font-weight: 700; font-size: 0.85rem; margin-bottom: 6px;}
  .vs-team {display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; margin: 4px 0 8px;}
  .vs-member {display: flex; align-items: center; gap: 12px; border: 1px solid rgba(127,127,127,0.25); border-radius: 12px;
              padding: 12px 14px;}
  .vs-member .a {flex: none; width: 44px; height: 44px; border-radius: 50%; background: #141414; color: #fff;
                 display: grid; place-items: center; font-weight: 800; font-size: 1.1rem; box-shadow: inset 0 0 0 2px #ff4d4d;}
  .vs-member b {font-size: 1.02rem;}
  .vs-member .r {font-size: 0.8rem; opacity: 0.7;}
  .vs-tagline {font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 18px 0 6px;}
  .vs-tagline span {color: #cd040b;}
</style>
"""


def _md(text: str) -> str:
    """Escape $ for Streamlit markdown, where a pair of dollar signs starts a formula."""
    return text.replace("$", "\\$")


def _band(kicker: str, title_html: str, text: str, stats: list[tuple[str, str]] | None = None) -> None:
    row = "".join(f"<div><b>{S.esc(n)}</b><span>{S.esc(label)}</span></div>" for n, label in stats or [])
    st.markdown(CSS + f'<div class="vs-band"><div class="k">{S.esc(kicker)}</div><div class="t">{title_html}</div>'
                f'<div class="p">{S.esc(text)}</div>' + (f'<div class="stats">{row}</div>' if row else "") + "</div>",
                unsafe_allow_html=True)


def _cards(items: list[tuple], columns: int) -> None:
    """Bordered cards: (icon, title, text) or (title, text)."""
    cols = st.columns(columns)
    for i, item in enumerate(items):
        icon, title, text = item if len(item) == 3 else (None, *item)
        with cols[i % columns].container(border=True):
            st.markdown((f":material/{icon}: " if icon else "") + f"**{_md(title)}**")
            st.caption(_md(text))


def _steps(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for i, (title, text) in enumerate(items):
        with cols[i]:
            st.markdown(f'<div class="vs-step">{i + 1}</div>', unsafe_allow_html=True)
            st.markdown(f"**{_md(title)}**")
            st.caption(_md(text))


def _team() -> None:
    cards = "".join(f'<div class="vs-member"><div class="a">{S.esc(n[:1])}</div><div><b>{S.esc(n)}</b>'
                    + (f'<div class="r">{S.esc(r)}</div>' if r else "") + "</div></div>" for n, r in C.TEAM)
    st.markdown(CSS + f'<div class="vs-team">{cards}</div>', unsafe_allow_html=True)


def team_line() -> str:
    names = [n for n, _ in C.TEAM]
    return "Built by " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)) + "."


def _link(key: str, label: str, icon: str) -> None:
    if key in LINKS:
        st.page_link(LINKS[key], label=label, icon=icon)


def _money() -> dict:
    hours, dollars = impact.transfer_savings(**impact.BRIEF)
    dev, biz, build = impact.build_cost()
    full = hours * impact.FULLY_LOADED_HOURLY_COST
    return {"hours": hours, "dollars": dollars, "dev": dev, "biz": biz, "build": build, "full": full,
            "payback": impact.payback_days(dollars), "payback_full": impact.payback_days(full),
            "fraud": impact.fraud_avoided(**impact.FRAUD_WHAT_IF)}


def about() -> None:
    S.header("About <span>Shield</span>", "What Shield is, why it matters, and how it decides.")
    m = _money()
    _band(C.ABOUT_KICKER, "Who is <span>really</span> calling?", C.ABOUT_INTRO,
          [(n, label) for n, label in C.PROOF[:2]] + [(f"${m['dollars'] / 1e6:.2f}M", "a year in agent time")])
    c = st.columns([1, 4], vertical_alignment="center")
    if c[0].button("Live demo", icon=":material/play_circle:", type="primary", key="live-demo") and "copilot" in LINKS:
        st.switch_page(LINKS["copilot"])
    c[1].markdown("**Pick a call and press Play:** you hear the caller and watch Shield score the risk, explain why and "
                  "tell the agent what to do, turn by turn.")
    _link("how", "How it works in detail", ":material/account_tree:")

    st.markdown("#### The problem: agents can't hear the difference")
    _cards(C.THREATS, 3)
    st.info(_md(C.TRANSFER_PROBLEM), icon=":material/swap_calls:")

    st.markdown("#### How it works")
    _steps(C.FLOW)
    st.markdown("| Signal | What it asks | How |\n|---|---|---|\n" +
                "\n".join(f"| **{n}** | {q} | {h} |" for n, q, h in C.SIGNALS))

    st.markdown("#### Two blind spots, one shield")
    _cards(C.BLIND_SPOTS, 2)

    st.markdown("#### Shield advises. People decide.")
    _cards(C.HUMAN_IN_THE_LOOP, 5)
    st.caption(_md(C.AI_ASSISTANTS))

    st.markdown("#### Measured, not claimed")
    k = st.columns(len(C.PROOF))
    for col, (n, label) in zip(k, C.PROOF):
        col.metric(label, n, border=True)
    st.caption(_md(C.PROOF_LIMITS))
    _link("evaluation", "See the model evaluation", ":material/monitoring:")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Built with")
        st.markdown(_md(C.BUILT_WITH))
    with right:
        st.markdown("#### Data and ethics")
        st.markdown(_md(C.DATA_ETHICS))
    if C.TEAM:
        st.markdown("#### Team")
        _team()


def vision() -> None:
    S.header("Our <span>vision</span>", "Where Shield goes after the pilot.")
    _band("Vision", S.esc(C.VISION_HEADLINE), C.VISION_INTRO)
    _cards(C.PILLARS, 2)

    st.markdown("#### Roadmap: from one queue to every call")
    _steps(C.ROADMAP)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### What the pilot needs")
        st.markdown("\n".join(f"- :material/check_circle: {n}" for n in C.PILOT_NEEDS))
    with right:
        st.markdown("#### Next in the product")
        st.markdown("\n".join(f"- **{t}:** {x}" for t, x in C.NEXT_IN_PRODUCT))
    st.markdown(f'<div class="vs-tagline">{S.esc(C.TAGLINE[0])} <span>{S.esc(C.TAGLINE[1])}</span></div>',
                unsafe_allow_html=True)
    if C.TEAM:
        st.caption(team_line())


def value() -> None:
    S.header("The <span>value</span>", "What Shield costs, what it returns, and the assumptions behind both.")
    m = _money()
    _band("Value", S.esc(C.VALUE_HEADLINE), C.VALUE_INTRO)
    k = st.columns(4)
    k[0].metric("Setup cost (one time, internal hours)", f"${m['build'] / 1e3:,.0f}k", border=True,
                help=_md(f"A {impact.BUILD_SIZE} effort: {m['dev']} developer/IT hours × "
                         f"${impact.RATE_CARD['developer']:.0f} + {m['biz']} business/SME hours × "
                         f"${impact.RATE_CARD['business']:.0f}, the sprint's rate card."))
    k[1].metric("Running cost (GCP, per year)", f"${impact.RUN_COST_PER_YEAR:,.0f}", border=True,
                help="Cloud infrastructure on GCP for the open-source models; no per-call license fees. Internal hours "
                     "are in the setup cost.")
    k[2].metric("Back, a year", f"${m['dollars'] / 1e6:.2f}M", border=True,
                help=_md(f"{m['hours']:,.0f} agent hours at the brief's ${impact.BRIEF['hourly_cost']:.0f}/hour."))
    k[3].metric("Payback", impact.payback_label(m["payback"]).replace("about ", "~"), border=True,
                help="Build cost ÷ daily savings after running cost.")
    u = st.columns(2)
    u[0].metric(f"Upside: at a fully loaded ${impact.FULLY_LOADED_HOURLY_COST:.0f}/hour", f"${m['full'] / 1e6:.1f}M a year",
                border=True, help=f"Same hours; payback {impact.payback_label(m['payback_full'])}.")
    w = impact.FRAUD_WHAT_IF
    u[1].metric("Upside: fraud avoided (illustrative)", f"${m['fraud'] / 1e6:.1f}M a year", border=True,
                help=_md(f"Mock inputs: {w['attempts_per_month']:,} bot or clone calls a month × ${w['loss_per_attempt']}, "
                         f"caught {w['baseline_catch_rate']:.0%} → {w['catch_rate']:.0%}. Not in the payback."))

    st.markdown("#### Where the value comes from")
    _cards(C.VALUE_STREAMS, 4)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("#### What it takes to build")
        rows = "\n".join(f"| {n} | {d or '-'} | {b or '-'} | ${d * impact.RATE_CARD['developer'] + b * impact.RATE_CARD['business']:,.0f} |"
                         for n, d, b in impact.BUILD_PACKAGES)
        st.markdown(_md("| Work package | Dev/IT h | SME h | Cost |\n|---|---:|---:|---:|\n" + rows +
                        f"\n| **Total: {impact.BUILD_SIZE} effort** | **{m['dev']}** | **{m['biz'] or '-'}** | **${m['build']:,.0f}** |"))
    with right:
        st.markdown("#### Even at a fraction of the volume")
        rows = "\n".join(f"| {s:.0%} | ${m['dollars'] * s / 1e6:.2f}M | {impact.payback_label(impact.payback_days(m['dollars'] * s))} |"
                         for s in (1.0, 0.75, 0.5, 0.25))
        st.markdown(_md("| Share of transfers Shield verifies | Back a year | Payback |\n|---|---:|---:|\n" + rows))
        _link("impact", "Change any assumption in the calculator", ":material/calculate:")
    st.caption(_md(C.VALUE_NOTES))
    if C.TEAM:
        st.caption(team_line())
