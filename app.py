"""Deepfake Voice Shield - agent co-pilot dashboard.

    streamlit run app.py
"""
import streamlit as st

from ui import about, copilot, evaluation, impact, product, try_voice
from ui import style as S

st.set_page_config(page_title="Deepfake Voice Shield", page_icon="🛡️", layout="wide")

copilot_page = st.Page(copilot.render, title="Call co-pilot", icon=":material/support_agent:", url_path="copilot",
                       default=True)
evaluation_page = st.Page(evaluation.render, title="Model evaluation", icon=":material/monitoring:", url_path="evaluation")
impact_page = st.Page(impact.render, title="Business impact", icon=":material/payments:", url_path="impact")
how_page = st.Page(about.render, title="How it works", icon=":material/account_tree:", url_path="how-it-works")
product_pages = [
    st.Page(product.about, title="About", icon=":material/info:", url_path="about"),
    st.Page(product.vision, title="Vision", icon=":material/rocket_launch:", url_path="vision"),
    st.Page(product.value, title="Value", icon=":material/trending_up:", url_path="value"),
]
product.LINKS.update(copilot=copilot_page, evaluation=evaluation_page, impact=impact_page, how=how_page)

nav = st.navigation({
    "Live demo": [copilot_page, st.Page(try_voice.render, title="Try your voice", icon=":material/mic:", url_path="try"),
                  evaluation_page, impact_page, how_page],
    "The product": product_pages,
})
S.top_nav(product_pages)
nav.run()
