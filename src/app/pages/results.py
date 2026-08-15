"""Página de resultados - Visualização e exportação."""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.app.theme import CLASS_COLORS, CLASS_NAMES

CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _pred_map_to_image(pred_map: np.ndarray) -> Image.Image:
    """Converte o mapa de classificação para imagem RGB colorida."""
    h, w = pred_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for name, idx in CLASS_INDEX.items():
        color = CLASS_COLORS.get(name, "#000000")
        hex_color = color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        rgb[pred_map == idx] = [r, g, b]

    return Image.fromarray(rgb, mode="RGB")


def _apply_classification_overlay(rgb_image: np.ndarray, pred_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Aplica a classificação sobre a imagem RGB com transparência."""
    h, w = pred_map.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for name, idx in CLASS_INDEX.items():
        color = CLASS_COLORS.get(name, "#000000")
        hex_color = color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        overlay[pred_map == idx] = [r, g, b]

    rgb_normalized = rgb_image.astype(np.float32)
    if rgb_normalized.max() > 1:
        rgb_normalized = rgb_normalized / 255.0

    overlay_normalized = overlay.astype(np.float32) / 255.0
    blended = (rgb_normalized * (1 - alpha) + overlay_normalized * alpha).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")


def _normalize_rgb(rgb_data: np.ndarray) -> np.ndarray:
    """Normaliza dados RGB para exibição."""
    rgb = rgb_data.copy()
    if rgb.max() > 255:
        rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255, 0, 255).astype(np.uint8)
    elif rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    return rgb


def _compute_areas(pred_map: np.ndarray) -> dict:
    """Calcula áreas por classe a partir do mapa de classificação."""
    total_pixels = pred_map.size
    areas = {}
    for name, idx in CLASS_INDEX.items():
        count = int(np.sum(pred_map == idx))
        areas[name] = {
            "pixels": count,
            "percent": count / total_pixels * 100 if total_pixels else 0,
        }
    return areas


def render_results():
    st.title("📊 Resultados da Classificação")
    st.markdown("---")

    if "last_classification" not in st.session_state:
        st.warning("⚠️ Nenhuma classificação realizada ainda.")
        st.info("Vá para a página 'Classificar' para iniciar.")
        return

    classification = st.session_state.last_classification

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Satélite", classification["satellite"].upper())
    with col2:
        st.metric("Classificador", classification["classifier"].upper())
    with col3:
        st.metric("Imagens", classification["num_images"])

    st.markdown("---")
    st.subheader("Mapa de Classificação")

    pred_map = classification.get("pred_map")
    meta = classification.get("meta")
    rgb_image = classification.get("rgb_image")

    if pred_map is None:
        st.info(
            "Classificação não disponível. Treine o modelo e execute a classificação."
        )
    else:
        st.success("✅ Resultado pronto.")

        view_mode = st.radio(
            "Visualização",
            ["📷 Imagem Real", "🎨 Classificação", "🔀 Sobreposta"],
            horizontal=True,
        )

        if view_mode == "📷 Imagem Real" and rgb_image is not None:
            st.image(_normalize_rgb(rgb_image), use_container_width=True, caption="Imagem real do satélite")
        elif view_mode == "🎨 Classificação":
            st.image(_pred_map_to_image(pred_map), use_container_width=True, caption="Mapa temático")
        elif view_mode == "🔀 Sobreposta" and rgb_image is not None:
            alpha = st.slider("Transparência da classificação", 0.0, 1.0, 0.5)
            st.image(
                _apply_classification_overlay(rgb_image, pred_map, alpha=alpha),
                use_container_width=True,
                caption="Classificação sobre a imagem real",
            )
        else:
            st.image(_pred_map_to_image(pred_map), use_container_width=True)

        areas = _compute_areas(pred_map)
        df = pd.DataFrame(
            [
                {
                    "Classe": name,
                    "Pixels": data["pixels"],
                    "Proporção (%)": round(data["percent"], 2),
                }
                for name, data in areas.items()
            ]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df.set_index("Classe")["Proporção (%)"])
        with col2:
            st.bar_chart(df.set_index("Classe")["Proporção (%)"], horizontal=True)

        st.subheader("Detalhamento")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Análise de Áreas Ociosas")
        st.markdown(
            """
        **Áreas identificadas como potenciais para arrendamento:**

        - **Solo Exposto:** Áreas sem cobertura vegetal, podem indicar terras disponíveis
        - **Pastagem degradada:** Áreas com NDVI baixo dentro de pastagens
        """
        )

        st.markdown("---")
        st.subheader("📥 Exportar Resultados")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Baixar Relatório (CSV)", use_container_width=True):
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Salvar CSV",
                    csv,
                    "resultado_classificacao.csv",
                    "text/csv",
                    use_container_width=True,
                )
        with col2:
            if st.button("🗺️ Baixar Mapa (PNG)", use_container_width=True):
                buf = io.BytesIO()
                _pred_map_to_image(pred_map).save(buf, format="PNG")
                st.download_button(
                    "Salvar PNG",
                    buf.getvalue(),
                    "mapa_classificacao.png",
                    "image/png",
                    use_container_width=True,
                )
