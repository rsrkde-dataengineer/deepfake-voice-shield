"""Deepfake Voice Shield - agent co-pilot dashboard.

    streamlit run app.py
"""
import streamlit as st

from ui import about, copilot, evaluation, impact, try_voice

st.set_page_config(page_title="Deepfake Voice Shield", page_icon="🛡️", layout="wide")

pages = [
    st.Page(copilot.render, title="Call co-pilot", icon=":material/support_agent:", url_path="copilot", default=True),
    st.Page(try_voice.render, title="Try your voice", icon=":material/mic:", url_path="try"),
    st.Page(evaluation.render, title="Model evaluation", icon=":material/monitoring:", url_path="evaluation"),
    st.Page(impact.render, title="Business impact", icon=":material/payments:", url_path="impact"),
    st.Page(about.render, title="How it works", icon=":material/account_tree:", url_path="how-it-works"),
]
st.navigation(pages).run()
