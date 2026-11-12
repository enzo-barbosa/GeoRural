"""Página de classificação - Desenho de polígono e processamento."""

import os
import tempfile

import ee
import folium
import numpy as np
import requests
import streamlit as st
from streamlit_folium import st_folium

from src.app.theme import CLASS_COLORS
from src.classifiers.random_forest import load_model as load_rf
from src.classifiers.random_forest import predict_random_forest as predict_rf_model
from src.classifiers.svm import load_model as load_svm
from src.classifiers.svm import predict_svm as predict_svm_model
from src.classifiers.unet_cnn import build_unet, predict_unet
from src.classifiers.unet_cnn import load_model as load_unet
from src.config import NUM_CLASSES
from src.gee.collector import (
    get_composite_image,
    polygon_to_ee_geometry,
    search_images,
)
from src.gee.preprocessor import preprocess_image
from src.geo.raster_io import read_geotiff
from src.processing.features import extract_features_ee, get_feature_bands

MODEL_PATHS = {
    "random_forest": "models/brazil_rf.pkl",
    "svm": "models/brazil_svm.pkl",
    "unet_cnn": "models/brazil_unet.pth",
}


def _load_classifier(classifier: str):
    if classifier == "random_forest":
        return load_rf(MODEL_PATHS["random_forest"])
    if classifier == "svm":
        return load_svm(MODEL_PATHS["svm"])
    if classifier == "unet_cnn":
        model = build_unet(input_shape=(64, 64, 9), num_classes=NUM_CLASSES)
        return load_unet(model, MODEL_PATHS["unet_cnn"])
    raise ValueError(f"Classificador não suportado: {classifier}")


def _download_ee_image(image: ee.Image, geometry, scale: int = 10, bandas: list | None = None):
    """Baixa uma imagem GEE para array numpy."""
    if bandas:
        image = image.select(bandas)
    url = image.getDownloadURL(
        {"region": geometry, "scale": scale, "format": "GEO_TIFF", "maxPixels": 1e9}
    )

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        tmp.write(response.content)
        tmp_path = tmp.name

    data, meta = read_geotiff(tmp_path)
    os.unlink(tmp_path)
    return data, meta


def _get_rgb_composite(composite, satellite):
    """Gera composição RGB real para visualização."""
    if satellite == "sentinel2":
        rgb = composite.select(["B4", "B3", "B2"])
    else:
        rgb = composite.select(["SR_B4", "SR_B3", "SR_B2"])
    return rgb


def _classify_area_real(composite, satellite, classifier):
    """Aplica o modelo treinado na imagem e retorna o mapa de classificação e imagem RGB real."""
    scale = 10 if satellite == "sentinel2" else 30
    rgb_img = _get_rgb_composite(composite, satellite)
    rgb_data, _ = _download_ee_image(rgb_img, composite.geometry(), scale=scale)

    data, meta_all = _download_ee_image(
        composite.select(get_feature_bands(satellite)), composite.geometry(), scale=scale
    )

    if data is None or data.size == 0 or rgb_data is None or rgb_data.size == 0:
        return None, None, None

    image = np.moveaxis(data, 0, -1)
    h, w = image.shape[:2]

    if classifier == "random_forest":
        model = load_rf(MODEL_PATHS["random_forest"])
        flat = image.reshape(-1, image.shape[-1])
        pred = predict_rf_model(model, flat)
        pred_map = pred.reshape(h, w)
    elif classifier == "svm":
        model = load_svm(MODEL_PATHS["svm"])
        flat = image.reshape(-1, image.shape[-1])
        pred = predict_svm_model(model, flat)
        pred_map = pred.reshape(h, w)
    elif classifier == "unet_cnn":
        model = build_unet(input_shape=(64, 64, 9), num_classes=NUM_CLASSES)
        model = load_unet(model, MODEL_PATHS["unet_cnn"])
        pred_map = predict_unet(model, image, patch_size=64, stride=32)
    else:
        return None, None, None

    rgb = np.moveaxis(rgb_data, 0, -1)
    return pred_map, meta_all, rgb


def _get_status_badge(model_loaded: bool) -> str:
    if model_loaded:
        return '<span class="status-badge success">Modelo pronto</span>'
    return '<span class="status-badge warning">Modelo não treinado</span>'


def render_classify():
    st.title("🗺️ Classificação de Uso do Solo")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1. Defina a Área de Interesse")

        m = folium.Map(location=[-14.235, -51.925], zoom_start=5, tiles="OpenStreetMap")

        from folium.plugins import Draw

        draw = Draw(
            draw_options={
                "polyline": False,
                "rectangle": True,
                "circle": False,
                "circlemarker": False,
                "marker": False,
            },
            edit_options={"edit": True},
        )
        draw.add_to(m)

        output = st_folium(m, width=700, height=500)

        polygon_data = None
        if output and output.get("all_drawings"):
            drawings = output["all_drawings"]
            if drawings:
                polygon_data = drawings[0]["geometry"]

        if polygon_data:
            st.success("✅ Área definida! Ajuste as configurações ao lado.")
        else:
            st.warning("⚠️ Desenhe uma área no mapa")

    with col2:
        st.subheader("2. Configurações")

        satellite = st.selectbox(
            "Satélite",
            ["sentinel2", "landsat8"],
            format_func=lambda x: (
                "Sentinel-2 (10m)" if x == "sentinel2" else "Landsat 8 (30m)"
            ),
        )

        classifier = st.selectbox(
            "Classificador",
            ["random_forest", "svm", "unet_cnn"],
            format_func=lambda x: {
                "random_forest": "Random Forest",
                "svm": "SVM",
                "unet_cnn": "U-Net CNN",
            }[x],
        )

        st.markdown("**Período de imagens:**")
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Data inicial", value=__import__("datetime").date(2023, 1, 1)
            )
        with col_end:
            end_date = st.date_input(
                "Data final", value=__import__("datetime").date(2023, 12, 31)
            )

        max_cloud = st.slider("Máximo de cobertura de nuvens (%)", 0, 100, 80)

        st.markdown("---")
        st.markdown("**Legenda das classes:**")
        legend_items = [
            ("Vegetação Nativa", CLASS_COLORS.get("Vegetação Nativa", "#1a5276")),
            ("Pastagem", CLASS_COLORS.get("Pastagem", "#0E6B3D")),
            ("Agricultura", CLASS_COLORS.get("Agricultura", "#2ecc71")),
            ("Solo Exposto", CLASS_COLORS.get("Solo Exposto", "#f39c12")),
            ("Área Construída", CLASS_COLORS.get("Área Construída", "#e74c3c")),
        ]

        for name, color in legend_items:
            st.markdown(
                f"""
            <div class="legend-item">
                <div class="legend-color" style="background-color: {color};"></div>
                <div style="font-size: 0.9rem;">{name}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("🚀 Classificar", type="primary", disabled=not polygon_data):
            classify_area(
                polygon_data,
                satellite,
                classifier,
                str(start_date),
                str(end_date),
                max_cloud,
            )


def classify_area(polygon_data, satellite, classifier, start_date, end_date, max_cloud):
    with st.spinner("🔄 Processando..."):
        geometry = polygon_to_ee_geometry(polygon_data["coordinates"][0])

        collection = search_images(
            geometry, start_date, end_date, satellite, max_cloud
        )

        num_images = collection.size().getInfo()
        st.info(f"📡 {num_images} imagens encontradas")

        if num_images == 0:
            st.error("❌ Nenhuma imagem encontrada para os critérios selecionados.")
            return

        composite = get_composite_image(collection, geometry)
        composite = preprocess_image(composite, satellite)
        composite = extract_features_ee(composite, satellite)

        model_path = MODEL_PATHS.get(classifier)
        model_loaded = False
        pred_map = None
        meta = None
        rgb_image = None
        if model_path and os.path.exists(model_path):
            try:
                pred_map, meta, rgb_image = _classify_area_real(composite, satellite, classifier)
                model_loaded = True
            except (OSError, RuntimeError, ValueError) as exc:
                st.warning(f"⚠️ Falha ao carregar o modelo: {exc}")

        st.session_state.last_classification = {
            "satellite": satellite,
            "classifier": classifier,
            "start_date": start_date,
            "end_date": end_date,
            "num_images": num_images,
            "geometry": polygon_data,
            "model_loaded": model_loaded,
            "pred_map": pred_map,
            "meta": meta,
            "rgb_image": rgb_image,
        }

        if model_loaded and pred_map is not None:
            st.success("✅ Classificação real concluída! Veja os resultados.")
        elif model_loaded:
            st.success("✅ Modelo carregado. Integração pronta para classificação real.")
        else:
            st.warning(
                "⚠️ Modelo ainda não treinado. "
                "Rode o notebook 03 ou 04 para gerar o modelo antes de classificar."
            )
