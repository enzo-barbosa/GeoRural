"""Core classification service logic."""

import json
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
from backend.app.dependencies.models import check_model_exists, get_cached_model
from backend.app.services.limits import area_limit_message, bands_per_download
from src.classifiers.crop_classifier import load_model as load_crop_model
from src.classifiers.crop_classifier import predict_crop_map
from src.classifiers.ensemble import (
    confidence_from_proba,
    load_weights,
    normalize_weights,
    soft_vote,
    weights_from_metrics,
)
from src.classifiers.unet_cnn import predict_unet_proba
from src.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_NAMES_GLOBAL,
    CROP_COLORS,
    CROP_NAMES,
    NUM_CLASSES,
    SLOPE_CLASS_COLORS,
    SLOPE_CLASS_NAMES,
)
from src.gee.collector import (
    get_composite_image,
    polygon_to_ee_geometry,
    search_images,
)
from src.gee.coverage import get_coverage, get_label_source
from src.gee.crop_features import get_crop_feature_image
from src.gee.preprocessor import preprocess_image
from src.gee.topography import (
    classify_slope,
    compute_terrain,
    dem_source_name,
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
from src.geo.postprocess import majority_filter
from src.geo.raster_io import read_geotiff
from src.geo.relief import elevation_legend, make_relief_rgb
from src.processing.features import extract_features_ee, get_feature_bands
from src.processing.local_features import add_local_indices_array


def _download_ee_image(
    image: ee.Image,
    geometry,
    scale: int,
    bandas: list | None = None,
    estimated_pixels: int | None = None,
):
    """Baixa uma imagem GEE como numpy array.

    O ``getDownloadURL`` do GEE aceita no máximo ~48 MB por requisição.
    Por isso as bandas são divididas em grupos pequenos: em áreas grandes,
    baixa-se uma banda por vez. Áreas que nem assim cabem são rejeitadas
    com uma mensagem clara.
    """
    if bandas:
        image = image.select(bandas)

    bands = image.bandNames().getInfo()
    if not bands:
        return None, None

    settings = get_settings()
    max_bytes = settings.MAX_DOWNLOAD_BYTES

    if estimated_pixels is None:
        area_m2 = geometry.area(maxError=1).getInfo()
        estimated_pixels = max(int(area_m2 / (scale * scale)), 1)

    bands_per_chunk = min(
        bands_per_download(estimated_pixels, max_bytes), len(bands)
    )

    chunks = []
    meta = None

    for i in range(0, len(bands), bands_per_chunk):
        group = bands[i : i + bands_per_chunk]
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


def _align_probabilities(
    probabilities: np.ndarray, classes, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Alinha colunas de probabilidade à ordem global de classes (0..N-1)."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    classes = [int(c) for c in np.asarray(classes).tolist()]

    if probabilities.shape[1] == num_classes and classes == list(range(num_classes)):
        return probabilities

    aligned = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
    for column, class_id in enumerate(classes):
        if 0 <= class_id < num_classes:
            aligned[:, class_id] = probabilities[:, column]
    return aligned


def _model_probabilities(model, features: np.ndarray):
    """Retorna (probabilidades, classes) para RF/SVM, com fallbacks."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(features), model.classes_
        except (AttributeError, ValueError):
            pass

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float32)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True), model.classes_

    predictions = model.predict(features).astype(int)
    probabilities = np.zeros((len(predictions), NUM_CLASSES), dtype=np.float32)
    probabilities[np.arange(len(predictions)), predictions] = 1.0
    return probabilities, model.classes_


def _domain_model_name(classifier_name: str, coverage: str) -> str:
    """Traduz o nome público para o modelo canônico do domínio."""
    prefix = "brazil" if coverage == "brazil" else "global"
    mapping = {
        "random_forest": f"{prefix}_rf",
        "svm": f"{prefix}_svm",
        "unet_cnn": f"{prefix}_unet",
    }
    return mapping.get(classifier_name, classifier_name)


def _fill_missing_weights(
    selected: dict[str, float], models: list[str]
) -> dict[str, float]:
    """Completa modelos sem peso com a média dos pesos conhecidos."""
    if not selected:
        return {}
    default = sum(selected.values()) / len(selected)
    return {name: selected.get(name, default) for name in models}


def _load_ensemble_weights(coverage: str, models: list[str]) -> dict[str, float]:
    """Pesos explícitos ou derivados do Kappa da validação espacial."""
    settings = get_settings()
    outputs = Path(settings.OUTPUTS_DIR)

    selected: dict[str, float] = {}

    explicit = load_weights(str(outputs / "ensemble_weights.json"))
    if explicit:
        selected = {name: explicit[name] for name in models if name in explicit}

    metrics_path = outputs / (
        "metrics_brazil.json" if coverage == "brazil" else "metrics_global.json"
    )
    if not selected and metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
        except (OSError, ValueError):
            metrics = {}

        alias = {
            "random_forest": f"{'brazil' if coverage == 'brazil' else 'global'}_rf",
            "svm": f"{'brazil' if coverage == 'brazil' else 'global'}_svm",
            "unet_cnn": "brazil_unet",
        }
        mapped = {
            alias.get(key, key): value
            for key, value in weights_from_metrics(metrics).items()
        }
        selected = {name: mapped[name] for name in models if name in mapped}

    # Peso da U-Net vem do mIoU/F1 salvo pelo treino no Colab.
    unet_metrics_path = outputs / "unet_metrics.json"
    if (
        "brazil_unet" in models
        and "brazil_unet" not in selected
        and unet_metrics_path.exists()
    ):
        try:
            with open(unet_metrics_path, encoding="utf-8") as f:
                unet_metrics = json.load(f)
        except (OSError, ValueError):
            unet_metrics = {}
        unet_weight = unet_metrics.get("miou") or unet_metrics.get("macro_f1")
        if unet_weight is not None:
            selected["brazil_unet"] = max(float(unet_weight), 0.0)

    if selected:
        return _fill_missing_weights(selected, models)

    return {}


def _load_sp_weights(models: list[str]) -> dict[str, float]:
    """Pesos do ensemble regional de São Paulo."""
    settings = get_settings()
    outputs = Path(settings.OUTPUTS_DIR)

    explicit = load_weights(str(outputs / "ensemble_sp_weights.json"))
    if explicit:
        selected = {name: explicit[name] for name in models if name in explicit}
        if selected:
            return _fill_missing_weights(selected, models)

    metrics_path = outputs / "metrics_sp.json"
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)
    except (OSError, ValueError):
        return {}
    weights = weights_from_metrics(
        {key: value for key, value in metrics.items() if isinstance(value, dict)}
    )
    selected = {name: weights[name] for name in models if name in weights}
    return _fill_missing_weights(selected, models)


def _predict_probability_maps(
    image: np.ndarray,
    satellite: str,
    coverage: str,
    classifier_name: str,
) -> tuple[np.ndarray, np.ndarray | None, list[str], dict[str, float]]:
    """Executa um modelo ou o ensemble e retorna mapa + probabilidades."""
    height, width = image.shape[:2]
    flat = image.reshape(-1, image.shape[-1])

    if classifier_name == "sp_ensemble":
        names = [
            name
            for name in ("sp_rf", "sp_svm")
            if check_model_exists(name)
        ]
        if (
            satellite == "sentinel2"
            and check_model_exists("brazil_unet")
        ):
            names.append("brazil_unet")
    elif classifier_name in {"auto", "ensemble"}:
        prefix = "brazil" if coverage == "brazil" else "global"
        names = [
            name
            for name in (f"{prefix}_rf", f"{prefix}_svm")
            if check_model_exists(name)
        ]
        if (
            coverage == "brazil"
            and satellite == "sentinel2"
            and check_model_exists("brazil_unet")
        ):
            names.append("brazil_unet")
    else:
        names = [_domain_model_name(classifier_name, coverage)]

    if not names:
        raise RuntimeError(
            "Nenhum modelo disponível para esta cobertura. Treine os modelos."
        )

    prefix = "brazil" if coverage == "brazil" else "global"
    if any(name.startswith(("brazil_", "global_")) and not name.startswith(prefix)
           for name in names):
        raise RuntimeError(
            "Modelo incompatível com a cobertura detectada "
            f"({coverage}). Use 'auto' ou um modelo do domínio correto."
        )
    if any(name.startswith("sp_") for name in names) and coverage != "brazil":
        raise RuntimeError(
            "O modelo regional SP está disponível apenas para o Brasil."
        )
    if any(name.startswith("sp_") for name in names) and satellite != "sentinel2":
        raise RuntimeError(
            "O modelo regional SP foi treinado apenas com Sentinel-2."
        )

    probabilities: dict[str, np.ndarray] = {}
    used: list[str] = []
    for name in names:
        model = get_cached_model(name)

        if name.endswith("_unet"):
            if satellite != "sentinel2":
                raise RuntimeError("A U-Net está disponível apenas para Sentinel-2.")
            n_channels = getattr(model, "n_channels", image.shape[-1])
            unet_input = image[..., :n_channels]
            try:
                proba = predict_unet_proba(
                    model,
                    unet_input,
                    patch_size=64,
                    stride=32,
                    num_classes=NUM_CLASSES,
                )
            except ValueError as exc:
                if len(names) == 1:
                    raise RuntimeError(
                        "Área muito pequena para a U-Net. "
                        "Selecione Random Forest/SVM ou aumente a área."
                    ) from exc
                print(f"[warn] U-Net ignorada nesta classificação: {exc}")
                continue
            probabilities[name] = proba.reshape(height, width, NUM_CLASSES)
        else:
            n_features = getattr(model, "n_features_in_", flat.shape[1])
            model_flat = flat[:, :n_features]
            proba, classes = _model_probabilities(model, model_flat)
            aligned = _align_probabilities(proba, classes)
            probabilities[name] = aligned.reshape(height, width, NUM_CLASSES)
        used.append(name)

    if not used:
        raise RuntimeError(
            "Nenhum modelo conseguiu processar a área selecionada."
        )

    if len(used) > 1:
        if any(name.startswith("sp_") for name in used):
            weights = _load_sp_weights(used)
        else:
            weights = _load_ensemble_weights(coverage, used)
    else:
        weights = {}
    combined = soft_vote(probabilities, weights)
    pred_map = np.argmax(combined, axis=-1).astype(np.uint8)

    normalized = {
        name: round(value, 4)
        for name, value in normalize_weights(used, weights).items()
    }
    return pred_map, combined, used, normalized


def _classify_area_real(
    composite,
    satellite: str,
    coverage: str,
    classifier_name: str,
    estimated_pixels: int | None = None,
):
    """Apply trained model(s) to image and return maps, RGB and metadata."""
    scale = 10 if satellite == "sentinel2" else 30

    rgb_img = _get_rgb_composite(composite, satellite)
    rgb_data, _ = _download_ee_image(
        rgb_img,
        composite.geometry(),
        scale=scale,
        estimated_pixels=estimated_pixels,
    )

    data, meta_all = _download_ee_image(
        composite.select(get_feature_bands(satellite)),
        composite.geometry(),
        scale=scale,
        estimated_pixels=estimated_pixels,
    )

    if data is None or data.size == 0 or rgb_data is None or rgb_data.size == 0:
        return None, None, None, None, [], {}

    image = np.moveaxis(data, 0, -1)
    image = add_local_indices_array(image)
    pred_map, probabilities, models_used, weights = _predict_probability_maps(
        image, satellite, coverage, classifier_name
    )

    rgb = np.moveaxis(rgb_data, 0, -1)
    return pred_map, probabilities, meta_all, rgb, models_used, weights


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
        if cid >= len(names) or item["pixels"] == 0:
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


def _rgba_from_rgb(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Adiciona transparência a um RGB fora da máscara do polígono."""
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def _compute_topography(geometry, pred_map, transform, crs, mask, job_id):
    """Executa a análise topográfica e retorna (stats, layers)."""
    scale = 30
    dem = get_dem(geometry)
    terrain = compute_terrain(dem)
    stats = elevation_stats(terrain, geometry, scale=scale)

    slope_img = classify_slope(terrain.select("slope"))
    slope_data, _ = _download_ee_image(
        slope_img, geometry, scale=scale, bandas=["slope_class"]
    )
    if slope_data is None:
        return None, []

    slope_map = slope_data[0] if slope_data.ndim == 3 else slope_data
    slope_map = nearest_resize(slope_map, pred_map.shape).astype(np.uint8)

    slope_metrics = compute_raster_metrics(
        slope_map, transform, crs, valid_mask=mask
    )
    topography = summarize(stats, slope_metrics)
    topography["dem_source"] = dem_source_name()
    topography["dem_resolution_m"] = scale

    layers = []

    slope_image = _layer_image(
        slope_map, SLOPE_CLASS_NAMES, SLOPE_CLASS_COLORS, mask
    )
    slope_url = _save_image(slope_image, job_id, "slope")
    layers.append(
        {
            "name": "slope",
            "label": "Declividade / Relevo",
            "image_url": slope_url,
            "legend": _legend(SLOPE_CLASS_NAMES, SLOPE_CLASS_COLORS),
        }
    )

    # Mapa de relevo: hipsometria + hillshade, exibido junto dos dados.
    relief_data, _ = _download_ee_image(
        terrain.select(["elevation", "hillshade"]),
        geometry,
        scale=scale,
    )
    if relief_data is not None and relief_data.shape[0] >= 2:
        elevation = nearest_resize(relief_data[0], pred_map.shape)
        hillshade = nearest_resize(relief_data[1], pred_map.shape)
        relief_rgb = make_relief_rgb(
            elevation,
            hillshade,
            vmin=stats.get("elevation_min"),
            vmax=stats.get("elevation_max"),
        )
        relief_url = _save_image(
            _rgba_from_rgb(relief_rgb, mask), job_id, "relief"
        )
        vmin = stats.get("elevation_min")
        vmax = stats.get("elevation_max")
        legend = (
            elevation_legend(float(vmin), float(vmax))
            if vmin is not None and vmax is not None
            else []
        )
        layers.append(
            {
                "name": "relief",
                "label": "Relevo (hipsometria + hillshade)",
                "image_url": relief_url,
                "legend": legend,
            }
        )

    return topography, layers


def _compute_crop(
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
    postprocess: bool = True,
) -> dict:
    """Main classification pipeline."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_id = f"{stamp}_{uuid.uuid4().hex[:8]}"

    geometry = polygon_to_ee_geometry(polygon_coords)

    poly = polygon_metrics(polygon_coords)
    area_ha = float(poly.get("area_ha", 0.0))
    scale = 10 if satellite == "sentinel2" else 30
    limit_message = area_limit_message(
        area_ha, scale, get_settings().MAX_CLASSIFY_PIXELS
    )
    if limit_message:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": limit_message,
        }
    estimated_pixels = int(area_ha * 10_000 / (scale * scale))

    try:
        coverage = get_coverage(polygon_coords)
    except Exception as exc:  # noqa: BLE001
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Falha ao detectar cobertura: {exc}",
        }

    label_source = get_label_source(coverage)
    crop_allowed = include_crop and coverage == "brazil"

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
        pred_map, probabilities, meta, rgb_image, models_used, weights = (
            _classify_area_real(
                composite,
                satellite,
                coverage,
                classifier_name,
                estimated_pixels=estimated_pixels,
            )
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

    postprocessed = False
    if postprocess:
        pred_map = majority_filter(
            pred_map, mask, kernel_size=3, num_classes=NUM_CLASSES
        )
        postprocessed = True

    class_names = CLASS_NAMES if coverage == "brazil" else CLASS_NAMES_GLOBAL

    classification_img = _layer_image(
        pred_map, class_names, CLASS_COLORS, mask
    )
    classification_url = _save_image(classification_img, job_id, "classification")

    rgb_url = None
    overlay_url = None
    if rgb_image is not None:
        rgb_normalized = _normalize_rgb(rgb_image)
        rgb_img = Image.fromarray(rgb_normalized, mode="RGB")
        rgb_url = _save_image(rgb_img, job_id, "rgb")

        class_rgb = colorize_by_name(pred_map, class_names, CLASS_COLORS)
        blended = blend(rgb_normalized, class_rgb, alpha=0.5)
        blended[~mask] = rgb_normalized[~mask]
        overlay_img = Image.fromarray(blended, mode="RGB")
        overlay_url = _save_image(overlay_img, job_id, "overlay")

    # Medidas geométricas reais
    raster_metrics = compute_raster_metrics(
        pred_map, transform, crs, valid_mask=mask
    )
    class_areas = _class_areas(raster_metrics, class_names)
    total_area_ha = poly.get("area_ha") or raster_metrics["total_area_ha"]
    measurements = {
        "total_area_ha": round(float(total_area_ha), 4),
        "perimeter_m": round(float(poly.get("perimeter_m", 0.0)), 2),
        "total_pixels": int(raster_metrics["total_pixels"]),
        "parcels": int(sum(c["parcels"] for c in class_areas)),
    }

    confidence_mean = None
    if probabilities is not None and mask.any():
        confidence = confidence_from_proba(probabilities)
        confidence_mean = round(float(confidence[mask].mean()), 4)

    layers = [
        {
            "name": "classification",
            "label": "Uso e Cobertura do Solo",
            "image_url": classification_url,
            "legend": _legend(class_names, CLASS_COLORS),
        }
    ]

    topography = None
    if include_topography:
        try:
            topography, topo_layers = _compute_topography(
                geometry, pred_map, transform, crs, mask, job_id
            )
            layers.extend(topo_layers)
        except Exception:  # noqa: BLE001
            topography = None

    crop_stats = None
    if crop_allowed:
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
        "coverage": coverage,
        "label_source": label_source,
        "models_used": models_used,
        "weights": weights,
        "confidence_mean": confidence_mean,
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
        "postprocessed": postprocessed,
        "layers": layers,
        "classification_image_url": classification_url,
        "rgb_image_url": rgb_url,
        "overlay_image_url": overlay_url,
        "bounds": bounds,
    }
