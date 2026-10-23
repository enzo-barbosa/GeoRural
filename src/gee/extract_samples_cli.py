"""Extrai amostras reais de treinamento a partir de imagem + MapBiomas."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from src.gee.auth import initialize_gee
from src.gee.collector import get_composite_image, polygon_to_ee_geometry, search_images
from src.gee.ground_truth import (
    extract_samples_gee,
    get_mapbiomas_layer,
    reclassify_mapbiomas,
)
from src.gee.preprocessor import preprocess_image
from src.processing.features import extract_features_ee


def run(polygon, start_date, end_date, satellite, max_cloud_pct, scale, seed, out_csv, samples_per_class):
    initialize_gee()

    geometry = polygon_to_ee_geometry(polygon)

    collection = search_images(
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
        satellite=satellite,
        max_cloud_pct=max_cloud_pct,
    )

    composite = get_composite_image(collection, geometry)
    composite = preprocess_image(composite, satellite=satellite)
    features = extract_features_ee(composite, satellite=satellite)

    gt = reclassify_mapbiomas(get_mapbiomas_layer(year=2023, geometry=geometry))

    training_points = extract_samples_gee(
        image=features,
        ground_truth=gt,
        region=geometry,
        scale=scale,
        seed=seed,
        samples_per_class=samples_per_class,
    )

    samples = training_points.getInfo()
    records = []
    for feature in samples.get("features", []):
        props = dict(feature["properties"])

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

        records.append(props)

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("Nenhuma amostra extraída. Verifique área, escala e datas.")

    if "class_id" not in df.columns:
        raise RuntimeError(f"Coluna 'class_id' ausente nas amostras. Colunas disponíveis: {df.columns.tolist()}")

    drop_cols = [c for c in df.columns if c.startswith(("QA", "ST_", "MSK_", "TCI_", "WVP", "AOT")) or c in {"SR_QA_AEROSOL"}]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    class_dist = df["class_id"].value_counts().sort_index()
    print("Distribuição das classes no CSV final:")
    for cid, count in class_dist.items():
        print(f"  classe {cid}: {count} amostras")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Amostras salvas em: {out_csv}")
    print(f"Total: {len(df)} amostras")
    print(f"Colunas: {df.columns.tolist()}")
    print(df.head())


def main():
    parser = argparse.ArgumentParser(description="Extrai amostras reais para treino")
    parser.add_argument("--out", default="data/samples/training_samples.csv")
    parser.add_argument("--satellite", default="sentinel2", choices=["sentinel2", "landsat8"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--max-cloud", type=int, default=20)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=500)

    args = parser.parse_args()

    polygon = [
        [-50.30, -20.25],
        [-50.20, -20.25],
        [-50.20, -20.30],
        [-50.30, -20.30],
        [-50.30, -20.25],
    ]

    run(
        polygon=polygon,
        start_date=args.start,
        end_date=args.end,
        satellite=args.satellite,
        max_cloud_pct=args.max_cloud,
        scale=args.scale,
        seed=args.seed,
        out_csv=args.out,
        samples_per_class=args.samples_per_class,
    )


if __name__ == "__main__":
    main()
