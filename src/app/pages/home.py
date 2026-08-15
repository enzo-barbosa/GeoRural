"""Página inicial da aplicação."""

import streamlit as st

from src.app.theme import CLASS_COLORS


def render_home():
    st.markdown(
        """
    <div class="main-header">
        <h1>🛰️ Geolocalização de Propriedades Rurais</h1>
        <p>Classifique o uso e cobertura do solo com imagens de satélite e aprendizado de máquina.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <h3>📡 Imagens de Satélite</h3>
            <p>Sentinel-2 e Landsat 8 via Google Earth Engine, com pré-processamento automático.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <h3>🤖 Classificadores</h3>
            <p>Random Forest, SVM e U-Net CNN para comparar acurácia e custo computacional.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="feature-card">
            <h3>🗺️ Interface Interativa</h3>
            <p>Desenhe a área, visualize resultados e identifique áreas ociosas no mapa.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Como usar")

    steps = [
        ("1", "Vá para <strong>Classificar</strong> na barra lateral."),
        (
            "2",
            "<strong>Desenhe um polígono</strong> no mapa definindo sua área de interesse.",
        ),
        (
            "3",
            "Selecione o <strong>satélite</strong> e o <strong>classificador</strong>.",
        ),
        ("4", "Clique em <strong>Classificar</strong> e aguarde o processamento."),
        ("5", "Visualize os <strong>resultados</strong> na página Resultados."),
    ]

    for number, text in steps:
        st.markdown(
            f"""
        <div class="step-indicator">
            <div class="step-number">{number}</div>
            <div class="step-content">{text}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Classes de Cobertura")

    legend_items = [
        ("Agricultura", "Lavouras cultivadas"),
        ("Pastagem", "Áreas de pasto"),
        ("Vegetação Nativa", "Mata, cerrado, floresta"),
        ("Solo Exposto", "Terra sem cobertura"),
        ("Área Construída", "Edificações, estradas"),
    ]

    for name, description in legend_items:
        color = CLASS_COLORS.get(name, "#888888")
        st.markdown(
            f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <div>
                <strong>{name}</strong><br/>
                <span style="color: #666; font-size: 0.9rem;">{description}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
    <div class="info-box">
        <strong>Dica:</strong> Para melhores resultados, desenhe áreas menores e evite período com muitas nuvens.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="footer">
        TCC - Geolocalização de Propriedades Rurais<br/>
        Centro Universitário de Fernandópolis
    </div>
    """,
        unsafe_allow_html=True,
    )
