"""Core classification service logic."""

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import requests
from PIL import Image

from backend.app.config import get_settings
from backend.app.dependencies.models import get_cached_model
from src.classifiers.crop_classifier import load_model as load_crop_model
from src.classifiers.crop_classifier import predict_crop_map
from src.classifiers.random_forest import predict_random_forest as predict_rf_model
from src.classifiers.svm import predict_svm as predict_svm_model
from src.classifiers.unet_cnn import predict_unet
from src.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    CROP_COLORS,
    CROP_NAMES,
    SLOPE_CLASS_COLORS,
    SLOPE_CLASS_NAMES,
)
from src.gee.collector import (
    get_composite_image,
    polygon_to_ee_geometry,
    search_images,
)
from src.gee.crop_features import get_crop_feature_image
from src.gee.preprocessor import preprocess_image
from src.gee.topography import (
    classify_slope,
    compute_terrain,
    elevation_stats,
    get_dem,
    summarize,
)
from src.geo.measurements import (
    compute_raster_metrics,
    polygon_mask,
    polygon_metrics,
    raster_bounds_lonlat,
)
from src.geo.overlay import (
    blend,
    colorize,
    colorize_by_name,
    nearest_resize,
)
from src.geo.raster_io import read_geotiff
from src.processing.features import extract_features_ee, get_feature_bands


def _download_ee_image(
    image: ee.Image, geometry, scale: int, bandas: list | None = None
):
    """Baixa uma imagem GEE como numpy array (em blocos de até 6 bandas)."""
    if bandas:
        image = image.select(bandas)

    bands = image.bandNames().getInfo()
    if not bands:
        return None, None

    chunks = []
    meta = None

    for i in range(0, len(bands), 6):
        group = bands[i : i + 6]
        url = image.select(group).getDownloadURL(
            {
                "region": geometry,
                "scale": scale,
                "format": "GEO_TIFF",
                "maxPixels": 1_000_000_000,
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp_path = tmp.name

        data, chunk_meta = read_geotiff(tmp_path)
        os.unlink(tmp_path)

        if meta is None:
            meta = chunk_meta
        chunks.append(data)

    data = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    return data, meta


def _get_rgb_composite(composite, satellite: str):
    """Generate RGB composite for visualization."""
    if satellite == "sentinel2":
        rgb = composite.select(["B4", "B3", "B2"])
    else:
        rgb = composite.select(["SR_B4", "SR_B3", "SR_B2"])
    return rgb


def _classify_area_real(composite, satellite: str, classifier_name: str, model):
    """Apply trained model to image and return classification map and RGB."""
    scale = 10 if satellite == "sentinel2" else 30

    rgb_img = _get_rgb_composite(composite, satellite)
    rgb_data, _ = _download_ee_image(rgb_img, composite.geometry(), scale=scale)

    data, meta_all = _download_ee_image(
        composite.select(get_feature_bands(satellite)),
        composite.geometry(),
        scale=scale,
    )

    if data is None or data.size == 0 or rgb_data is None or rgb_data.size == 0:
        return None, None, None

    image = np.moveaxis(data, 0, -1)
    h, w = image.shape[:2]

    if classifier_name == "random_forest":
        flat = image.reshape(-1, image.shape[-1])
        pred = predict_rf_model(model, flat)
        pred_map = pred.reshape(h, w)
    elif classifier_name == "svm":
        flat = image.reshape(-1, image.shape[-1])
        pred = predict_svm_model(model, flat)
        pred_map = pred.reshape(h, w)
    elif classifier_name == "unet_cnn":
        pred_map = predict_unet(model, image, patch_size=64, stride=32)
    else:
        return None, None, None

    rgb = np.moveaxis(rgb_data, 0, -1)
    return pred_map, meta_all, rgb


def _normalize_rgb(rgb_data: np.ndarray) -> np.ndarray:
    """Normalize RGB data for display."""
    rgb = rgb_data.copy()
    if rgb.max() > 255:
        rgb = np.clip(
            (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255, 0, 255
        ).astype(np.uint8)
    elif rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    return rgb.astype(np.uint8)


def _layer_image(class_map, names, colors, mask) -> Image.Image:
    """Gera PNG RGBA de uma camada, transparente fora do polígono."""
    if isinstance(colors, dict):
        rgb = colorize_by_name(class_map, names, colors)
    else:
        rgb = colorize(class_map, colors)

    alpha = np.where(mask, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _legend(names: list, colors) -> list[dict]:
    """Monta os itens de legenda (label + cor) de uma camada."""
    if isinstance(colors, dict):
        return [
            {"label": name, "color": colors.get(name, "#000000")}
            for name in names
        ]
    return [
        {"label": name, "color": colors[i]} for i, name in enumerate(names)
    ]


def _class_areas(metrics: dict, names: list) -> list[dict]:
    """Converte métricas do raster em lista por nome de classe."""
    result = []
    for item in metrics["classes"]:
        cid = item["class_id"]
        if cid >= len(names):
            continue
        result.append(
            {
                "class_name": names[cid],
                "area_ha": round(item["area_ha"], 4),
                "pixels": item["pixels"],
                "percent": round(item["percent"], 2),
                "parcels": item["parcels"],
            }
        )
    return result


def _save_image(image: Image.Image, job_id: str, suffix: str) -> str:
    """Save image and return URL path."""
    settings = get_settings()
    output_dir = Path(settings.OUTPUTS_DIR) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{suffix}.png"
    image.save(filepath)
    return f"/outputs/{job_id}/{suffix}.png"


def _compute_topography(
    geometry, pred_map, transform, crs, mask, job_id
):
    """Executa a análise topográfica e retorna (stats, layer)."""
    scale = 30
    dem = get_dem(geometry)
    terrain = compute_terrain(dem)
    stats = elevation_stats(terrain, geometry, scale=scale)

    slope_img = classify_slope(terrain.select("slope"))
    slope_data, _ = _download_ee_image(
        slope_img, geometry, scale=scale, bandas=["slope_class"]
    )
    if slope_data is None:
        return None, None

    slope_map = slope_data[0] if slope_data.ndim == 3 else slope_data
    slope_map = nearest_resize(slope_map, pred_map.shape).astype(np.uint8)

    slope_metrics = compute_raster_metrics(
        slope_map, transform, crs, valid_mask=mask
    )
    topography = summarize(stats, slope_metrics)

    slope_image = _layer_image(
        slope_map, SLOPE_CLASS_NAMES, SLOPE_CLASS_COLORS, mask
    )
    slope_url = _save_image(slope_image, job_id, "slope")

    layer = {
        "name": "slope",
        "label": "Declividade / Relevo",
        "image_url": slope_url,
        "legend": _legend(SLOPE_CLASS_NAMES, SLOPE_CLASS_COLORS),
    }
    return topography, layer


def _compute_crop(
    geometry, pred_map, transform, crs, mask, job_id, satellite, start_date,
    end_date, max_cloud,
):
    """Executa a classificação de tipo de cultivo e retorna (stats, layer)."""
    settings = get_settings()
    model_path = Path(settings.MODELS_DIR) / "crop_classifier.pkl"
    if not model_path.exists():
        return None, None

    scale = 10 if satellite == "sentinel2" else 30
    feature_img = get_crop_feature_image(
        geometry, start_date, end_date, satellite, max_cloud
    )
    crop_data, _ = _download_ee_image(feature_img, geometry, scale=scale)
    if crop_data is None:
        return None, None

    features = np.moveaxis(crop_data, 0, -1)
    features = nearest_resize(features, pred_map.shape)

    model = load_crop_model(str(model_path))
    crop_map = predict_crop_map(model, features)

    # O tipo de cultivo só faz sentido em áreas agrícolas ou de pastagem.
    crop_mask = mask & np.isin(pred_map, [1, 2])
    if not crop_mask.any():
        return None, None

    crop_metrics = compute_raster_metrics(
        crop_map, transform, crs, valid_mask=crop_mask
    )
    crop_stats = _class_areas(crop_metrics, CROP_NAMES)

    crop_image = _layer_image(crop_map, CROP_NAMES, CROP_COLORS, crop_mask)
    crop_url = _save_image(crop_image, job_id, "crop")

    layer = {
        "name": "crop",
        "label": "Tipo de Cultivo",
        "image_url": crop_url,
        "legend": _legend(CROP_NAMES, CROP_COLORS),
    }
    return crop_stats, layer


async def classify_area(
    polygon_coords: list[list[float]],
    satellite: str,
    classifier_name: str,
    start_date: str,
    end_date: str,
    max_cloud: int,
    include_topography: bool = True,
    include_crop: bool = True,
) -> dict:
    """Main classification pipeline."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_id = f"{stamp}_{uuid.uuid4().hex[:8]}"

    geometry = polygon_to_ee_geometry(polygon_coords)

    collection = search_images(geometry, start_date, end_date, satellite, max_cloud)
    num_images = collection.size().getInfo()

    if num_images == 0:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": "Nenhuma imagem encontrada para os critérios selecionados.",
        }

    composite = get_composite_image(collection, geometry)
    composite = preprocess_image(composite, satellite)
    composite = extract_features_ee(composite, satellite)

    try:
        model = get_cached_model(classifier_name)
    except Exception as exc:  # noqa: BLE001
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Falha ao carregar modelo: {exc}",
        }

    try:
        pred_map, meta, rgb_image = _classify_area_real(
            composite, satellite, classifier_name, model
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Falha na classificação: {exc}",
        }

    if pred_map is None:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": "Falha ao processar imagem - dados vazios.",
        }

    pred_map = np.asarray(pred_map)
    transform = meta["transform"]
    crs = meta["crs"]
    mask = polygon_mask(polygon_coords, transform, crs, pred_map.shape)
    bounds = raster_bounds_lonlat(transform, crs, pred_map.shape)

    classification_img = _layer_image(
        pred_map, CLASS_NAMES, CLASS_COLORS, mask
    )
    classification_url = _save_image(classification_img, job_id, "classification")

    rgb_url = None
    overlay_url = None
    if rgb_image is not None:
        rgb_normalized = _normalize_rgb(rgb_image)
        rgb_img = Image.fromarray(rgb_normalized, mode="RGB")
        rgb_url = _save_image(rgb_img, job_id, "rgb")

        class_rgb = colorize_by_name(pred_map, CLASS_NAMES, CLASS_COLORS)
        blended = blend(rgb_normalized, class_rgb, alpha=0.5)
        blended[~mask] = rgb_normalized[~mask]
        overlay_img = Image.fromarray(blended, mode="RGB")
        overlay_url = _save_image(overlay_img, job_id, "overlay")

    # Medidas geométricas reais
    raster_metrics = compute_raster_metrics(
        pred_map, transform, crs, valid_mask=mask
    )
    poly = polygon_metrics(polygon_coords)
    class_areas = _class_areas(raster_metrics, CLASS_NAMES)
    total_area_ha = poly.get("area_ha") or raster_metrics["total_area_ha"]
    measurements = {
        "total_area_ha": round(float(total_area_ha), 4),
        "perimeter_m": round(float(poly.get("perimeter_m", 0.0)), 2),
        "total_pixels": int(raster_metrics["total_pixels"]),
        "parcels": int(sum(c["parcels"] for c in class_areas)),
    }

    layers = [
        {
            "name": "classification",
            "label": "Uso e Cobertura do Solo",
            "image_url": classification_url,
            "legend": _legend(CLASS_NAMES, CLASS_COLORS),
        }
    ]

    topography = None
    if include_topography:
        try:
            topography, slope_layer = _compute_topography(
                geometry, pred_map, transform, crs, mask, job_id
            )
            if slope_layer:
                layers.append(slope_layer)
        except Exception:  # noqa: BLE001
            topography = None

    crop_stats = None
    if include_crop:
        try:
            crop_stats, crop_layer = _compute_crop(
                geometry,
                pred_map,
                transform,
                crs,
                mask,
                job_id,
                satellite,
                start_date,
                end_date,
                max_cloud,
            )
            if crop_layer:
                layers.append(crop_layer)
        except Exception:  # noqa: BLE001
            crop_stats = None

    return {
        "job_id": job_id,
        "status": "completed",
        "satellite": satellite,
        "classifier": classifier_name,
        "start_date": start_date,
        "end_date": end_date,
        "num_images": num_images,
        "stats": [
            {
                "class_name": c["class_name"],
                "pixels": c["pixels"],
                "percent": c["percent"],
            }
            for c in class_areas
        ],
        "measurements": measurements,
        "class_areas": class_areas,
        "topography": topography,
        "crop_stats": crop_stats,
        "layers": layers,
        "classification_image_url": classification_url,
        "rgb_image_url": rgb_url,
        "overlay_image_url": overlay_url,
        "bounds": bounds,
    }
