"""Componentes reutilizáveis da interface."""

import streamlit as st


def render_header(title: str, subtitle: str):
    """Renderiza um header estilizado."""
    st.markdown(
        f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Renderiza o footer da aplicação."""
    st.markdown(
        """
    <div class="footer">
        TCC - Geolocalização de Propriedades Rurais<br/>
        Centro Universitário de Fernandópolis
    </div>
    """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, icon: str = ""):
    """Renderiza um card de métrica estilizado."""
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def classification_legend():
    """Renderiza a legenda das classes de cobertura."""
    st.markdown(
        """
    <div class="legend-container">
        <strong>Legenda:</strong>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #2ecc71;"></div>
            <div>Agricultura</div>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #27ae60;"></div>
            <div>Pastagem</div>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #1a5276;"></div>
            <div>Vegetação Nativa</div>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #f39c12;"></div>
            <div>Solo Exposto</div>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #e74c3c;"></div>
            <div>Área Construída</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """Renderiza um badge de status."""
    css_class = (
        "success"
        if status == "success"
        else "warning"
        if status == "warning"
        else "error"
    )
    return f'<span class="status-badge {css_class}">{status}</span>'
