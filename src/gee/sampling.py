"""Amostragem multi-região/multi-ano para treino dos classificadores.

Cada região/ano é processado independentemente, com CSV parcial e manifesto
para permitir retomada. Os CSVs parciais são concatenados no final.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

from src.gee.collector import (
    get_composite_image,
    search_images,
)
from src.gee.crop_features import get_crop_feature_image
from src.gee.ground_truth import (
    extract_samples_gee,
    get_mapbiomas_layer,
    get_worldcover_layer,
    reclassify_mapbiomas,
    reclassify_mapbiomas_crops,
    reclassify_worldcover,
)
from src.gee.preprocessor import preprocess_image
from src.processing.features import extract_features_ee, get_feature_bands

DROP_PREFIXES = ("QA", "ST_", "MSK_", "TCI_", "WVP", "AOT", "SCL")


def build_feature_image(
    geometry,
    year: int,
    satellite: str = "sentinel2",
    max_cloud_pct: int = 80,
):
    """Monta a imagem de atributos anual de uma região.

    Returns:
        (features, num_images) ou (None, 0) se não houver imagens.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    collection = search_images(
        geometry, start_date, end_date, satellite, max_cloud_pct
    )
    num_images = collection.size().getInfo()
    if num_images == 0:
        return None, 0

    composite = get_composite_image(collection, geometry)
    composite = preprocess_image(composite, satellite)
    features = extract_features_ee(composite, satellite)
    features = features.select(get_feature_bands(satellite))
    return features, num_images


def feature_collection_to_dataframe(feature_collection, metadata: dict) -> pd.DataFrame:
    """Converte FeatureCollection em DataFrame, anexando metadados."""
    payload = feature_collection.getInfo()
    records = []

    for feature in payload.get("features", []):
        props = dict(feature.get("properties") or {})

        if "longitude" not in props or "latitude" not in props:
            geometry = feature.get("geometry") or {}
            coords = geometry.get("coordinates")
            if (
                isinstance(coords, (list, tuple))
                and len(coords) >= 2
                and isinstance(coords[0], (int, float))
            ):
                props["longitude"] = coords[0]
                props["latitude"] = coords[1]

        props.update(metadata)
        records.append(props)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    drop_cols = [
        c
        for c in df.columns
        if c.startswith(DROP_PREFIXES) or c == "SR_QA_AEROSOL"
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def extract_region_samples(
    region,
    year: int,
    label_source: str,
    satellite: str = "sentinel2",
    max_cloud_pct: int = 80,
    scale: int = 10,
    seed: int = 42,
    samples_per_class: int = 500,
    tile_scale: int = 4,
) -> tuple[pd.DataFrame, int]:
    """Extrai amostras de uso do solo de uma região/ano."""
    geometry = region.to_ee_geometry()
    features, num_images = build_feature_image(
        geometry, year, satellite, max_cloud_pct
    )
    if features is None:
        return pd.DataFrame(), 0

    if label_source == "mapbiomas":
        ground_truth = reclassify_mapbiomas(
            get_mapbiomas_layer(year=year, geometry=geometry)
        )
        label_year = year
    elif label_source == "worldcover":
        label_year = 2021
        ground_truth = reclassify_worldcover(
            get_worldcover_layer(year=label_year, geometry=geometry)
        )
    else:
        raise ValueError(f"label_source inválido: {label_source}")

    feature_collection = extract_samples_gee(
        image=features,
        ground_truth=ground_truth,
        region=geometry,
        scale=scale,
        seed=seed,
        samples_per_class=samples_per_class,
        class_band="class_id",
        tile_scale=tile_scale,
    )

    metadata = {
        "region_id": region.name,
        "biome": region.biome,
        "country": region.country,
        "domain": region.domain,
        "year": year,
        "label_year": label_year,
        "label_source": label_source,
    }
    df = feature_collection_to_dataframe(feature_collection, metadata)
    return df, num_images


def extract_crop_region_samples(
    region,
    year: int,
    satellite: str = "sentinel2",
    max_cloud_pct: int = 80,
    scale: int = 10,
    seed: int = 42,
    samples_per_crop: int = 500,
    tile_scale: int = 4,
) -> tuple[pd.DataFrame, int]:
    """Extrai amostras de tipo de cultivo (série NDVI/EVI) de uma região/ano."""
    geometry = region.to_ee_geometry()
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    crop_image = get_crop_feature_image(
        geometry, start_date, end_date, satellite, max_cloud_pct
    )
    ground_truth = reclassify_mapbiomas_crops(
        get_mapbiomas_layer(year=year, geometry=geometry)
    )

    feature_collection = extract_samples_gee(
        image=crop_image,
        ground_truth=ground_truth,
        region=geometry,
        scale=scale,
        seed=seed,
        samples_per_class=samples_per_crop,
        class_band="crop_id",
        tile_scale=tile_scale,
    )

    metadata = {
        "region_id": region.name,
        "biome": region.biome,
        "country": region.country,
        "domain": region.domain,
        "year": year,
        "label_source": "mapbiomas_crop",
    }
    df = feature_collection_to_dataframe(feature_collection, metadata)
    return df, 0


def update_manifest(path: str, entry: dict) -> None:
    """Adiciona/atualiza uma entrada no manifesto JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

    key = f"{entry.get('domain')}:{entry.get('region_id')}:{entry.get('year')}"
    data[key] = {
        **entry,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manifest(path: str) -> dict:
    """Carrega o manifesto (ou retorna vazio)."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_csvs(csv_paths: list[str], out_csv: str, dedupe: bool = True) -> pd.DataFrame:
    """Concatena CSVs parciais em um único arquivo."""
    frames = [
        pd.read_csv(path)
        for path in csv_paths
        if os.path.exists(path) and os.path.getsize(path) > 0
    ]
    if not frames:
        raise RuntimeError("Nenhum CSV parcial encontrado para concatenar.")

    df = pd.concat(frames, ignore_index=True)

    if dedupe and {"longitude", "latitude", "year"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["longitude", "latitude", "year"])

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
