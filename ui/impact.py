"""Page 4 - Business impact: the brief's savings model, with every assumption adjustable."""
from __future__ import annotations

import streamlit as st

from ui import style as S
from vshield import impact


def render() -> None:
    S.header("Business <span>impact</span>",
             "What skipping re-verification on transfers, and stopping bot and clone calls, is worth. "
             "Defaults reproduce the brief; change any assumption.")
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("##### Transfers")
        transfers = st.number_input("Transferred calls per year", 100_000, 20_000_000, impact.BRIEF["transfers_per_year"],
                                    step=50_000, help="3.65M is what the brief's 76,000 hours imply at 75 s per transfer.")
        seconds = st.slider("Re-verification time saved per transfer (s)", 30, 120, impact.BRIEF["seconds_saved"],
                            help="The brief: 60-90 seconds.")
        share = st.slider("Share of transferred callers Shield verifies", 50, 100, int(impact.BRIEF["share_verified"] * 100), 5,
                          format="%d%%", help="Callers that are stepped up or escalated still re-verify. "
                                              "100% = the brief's assumption.") / 100
        cost = st.number_input("Loaded agent cost per hour ($)", 10.0, 100.0, impact.BRIEF["hourly_cost"], step=1.0)
        st.markdown("##### Fraud (illustrative)")
        attempts = st.number_input("Bot and clone calls per month", 0, 100_000, 2_000, step=100,
                                   help="Mock figure - replace with the fraud team's count.")
        loss = st.number_input("Average loss per successful attempt ($)", 0, 5_000, 150, step=10,
                               help="Credits, waived fees or SIM-swap liability. Mock figure.")
        today = st.slider("Share caught today", 0, 100, 30, 5, format="%d%%", help="Mock figure.") / 100
        shield = st.slider("Share caught with Shield", 0, 100, 95, 5, format="%d%%",
                           help="Leaves room for newer generators than the ones in the evaluation.") / 100
    hours, dollars = impact.transfer_savings(transfers, seconds, share, cost)
    fraud = impact.fraud_avoided(attempts, loss, shield, today)
    with right:
        k = st.columns(2)
        k[0].metric("Agent hours saved per year", f"{hours:,.0f}", border=True)
        k[1].metric("Labour cost saved per year", f"${dollars / 1e6:,.2f}M", border=True)
        k = st.columns(2)
        k[0].metric("Full-time agents freed", f"{hours / 1_800:,.0f}", help="At 1,800 productive hours per agent-year.",
                    border=True)
        k[1].metric("Fraud losses avoided per year", f"${fraud / 1e6:,.2f}M", help="Illustrative: mock inputs.", border=True)
        usd = "\\$"   # a bare $ starts a LaTeX formula in Streamlit markdown
        st.markdown(f"""
**How it adds up**

- {transfers:,.0f} transfers × {share:.0%} verified × {seconds} s = **{hours:,.0f} agent hours** a year,
  × {usd}{cost:,.0f}/h = **{usd}{dollars:,.0f}**.
- The brief quotes **~76,000 hours and ~{usd}2.2M** a year at 60-90 s per transferred call; the defaults here
  reproduce it.
- Fraud: {attempts:,.0f} attempts a month × 12 × {usd}{loss:,.0f} × ({shield:.0%} − {today:.0%}) caught =
  **{usd}{fraud:,.0f}** a year (mock inputs - replace with real fraud data).
- Baseline to compare against: the Looker *chip_cs Handoffs* dashboard (transfer counts and re-verification time).
""")
        st.caption("Not counted: shorter average handle time on the first call, fewer SIM-swap liabilities, and "
                   "customer-experience gains from not repeating security questions.")
