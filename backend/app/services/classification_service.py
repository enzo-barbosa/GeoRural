"""Core classification service logic."""

import io
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import requests
from PIL import Image

from backend.app.config import get_settings
from backend.app.dependencies.models import get_cached_model
from src.classifiers.random_forest import predict_random_forest as predict_rf_model
from src.classifiers.svm import predict_svm as predict_svm_model
from src.classifiers.unet_cnn import build_unet, predict_unet
from src.gee.collector import (
    get_composite_image,
    polygon_to_ee_geometry,
    search_images,
)
from src.gee.preprocessor import preprocess_image
from src.processing.features import extract_features_ee, get_feature_bands
from src.geo.raster_io import read_geotiff


def _download_ee_image(image: ee.Image, geometry, scale: int, bandas: list | None = None):
    """Download a GEE image as numpy array."""
    if bandas:
        image = image.select(bandas)

    url = image.getDownloadURL(
        {"region": geometry, "scale": scale, "format": "GEO_TIFF", "maxPixels": 1_000_000_000}
    )

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        tmp.write(response.content)
        tmp_path = tmp.name

    data, meta = read_geotiff(tmp_path)
    os.unlink(tmp_path)
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

    # Download RGB for visualization
    rgb_img = _get_rgb_composite(composite, satellite)
    rgb_data, _ = _download_ee_image(rgb_img, composite.geometry(), scale=scale)

    # Download feature bands
    data, meta_all = _download_ee_image(
        composite.select(get_feature_bands(satellite)), composite.geometry(), scale=scale
    )

    if data is None or data.size == 0 or rgb_data is None or rgb_data.size == 0:
        return None, None, None

    # Reorganize to (height, width, channels)
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


def _pred_map_to_image(pred_map: np.ndarray) -> Image.Image:
    """Convert classification map to RGB image."""
    from src.config import CLASS_COLORS, CLASS_NAMES

    CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    h, w = pred_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for name, idx in CLASS_INDEX.items():
        color = CLASS_COLORS.get(name, "#000000")
        hex_color = color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb[pred_map == idx] = [r, g, b]

    return Image.fromarray(rgb, mode="RGB")


def _normalize_rgb(rgb_data: np.ndarray) -> np.ndarray:
    """Normalize RGB data for display."""
    rgb = rgb_data.copy()
    if rgb.max() > 255:
        rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255, 0, 255).astype(np.uint8)
    elif rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    return rgb


def _compute_areas(pred_map: np.ndarray) -> list[dict]:
    """Compute area statistics per class."""
    from src.config import CLASS_COLORS, CLASS_NAMES

    CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    total_pixels = pred_map.size
    stats = []

    for name, idx in CLASS_INDEX.items():
        count = int(np.sum(pred_map == idx))
        stats.append({
            "class_name": name,
            "pixels": count,
            "percent": round(count / total_pixels * 100, 2) if total_pixels else 0,
        })
    return stats


def _save_image(image: Image.Image, job_id: str, suffix: str) -> str:
    """Save image and return URL path."""
    settings = get_settings()
    output_dir = Path(settings.OUTPUTS_DIR) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{suffix}.png"
    image.save(filepath)
    return f"/outputs/{job_id}/{suffix}.png"


async def classify_area(
    polygon_coords: list[list[float]],
    satellite: str,
    classifier_name: str,
    start_date: str,
    end_date: str,
    max_cloud: int,
) -> dict:
    """Main classification pipeline."""
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Convert polygon to GEE geometry
    geometry = polygon_to_ee_geometry(polygon_coords)

    # Search images
    collection = search_images(geometry, start_date, end_date, satellite, max_cloud)
    num_images = collection.size().getInfo()

    if num_images == 0:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": "Nenhuma imagem encontrada para os critérios selecionados.",
        }

    # Create composite and preprocess
    composite = get_composite_image(collection, geometry)
    composite = preprocess_image(composite, satellite)
    composite = extract_features_ee(composite, satellite)

    # Load model
    try:
        model = get_cached_model(classifier_name)
    except Exception as exc:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Falha ao carregar modelo: {exc}",
        }

    # Classify
    try:
        pred_map, meta, rgb_image = _classify_area_real(composite, satellite, classifier_name, model)
    except Exception as exc:
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

    # Generate output images
    classification_img = _pred_map_to_image(pred_map)
    classification_url = _save_image(classification_img, job_id, "classification")

    rgb_url = None
    overlay_url = None
    if rgb_image is not None:
        rgb_normalized = _normalize_rgb(rgb_image)
        rgb_img = Image.fromarray(rgb_normalized, mode="RGB")
        rgb_url = _save_image(rgb_img, job_id, "rgb")

        # Create overlay
        from src.config import CLASS_COLORS, CLASS_NAMES
        CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        h, w = pred_map.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for name, idx in CLASS_INDEX.items():
            color = CLASS_COLORS.get(name, "#000000")
            hex_color = color.lstrip("#")
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            overlay[pred_map == idx] = [r, g, b]

        rgb_norm = rgb_normalized.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0
        blended = (rgb_norm * 0.5 + overlay_norm * 0.5) * 255
        blended = blended.astype(np.uint8)
        overlay_img = Image.fromarray(blended, mode="RGB")
        overlay_url = _save_image(overlay_img, job_id, "overlay")

    stats = _compute_areas(pred_map)

    return {
        "job_id": job_id,
        "status": "completed",
        "satellite": satellite,
        "classifier": classifier_name,
        "start_date": start_date,
        "end_date": end_date,
        "num_images": num_images,
        "stats": stats,
        "classification_image_url": classification_url,
        "rgb_image_url": rgb_url,
        "overlay_image_url": overlay_url,
    }