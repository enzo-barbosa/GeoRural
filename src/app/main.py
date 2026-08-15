"""Aplicação principal - Geolocalização de Propriedades Rurais."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.app.theme import get_css

st.set_page_config(page_title="Geolocalização Rural", page_icon="🛰️", layout="wide")

st.markdown(get_css(), unsafe_allow_html=True)

if "gee_initialized" not in st.session_state:
    from src.gee.auth import initialize_gee

    initialize_gee()
    st.session_state.gee_initialized = True

st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["🏠 Início", "🗺️ Classificar", "📊 Resultados"])

if page == "🏠 Início":
    from src.app.pages.home import render_home

    render_home()
elif page == "🗺️ Classificar":
    from src.app.pages.classify import render_classify

    render_classify()
elif page == "📊 Resultados":
    from src.app.pages.results import render_results

    render_results()
