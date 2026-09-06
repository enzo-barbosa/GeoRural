"""Construção do dataset completo (rasters) para treinamento da U-Net.

A U-Net é patch-based: precisa de uma imagem de satélite COMPLETA e do
máscara de ground truth COMPLETA (não apenas pixels esparsos como no CSV
de RF/SVM). Este script baixa ambos via Google Earth Engine para
data/processed/ no formato .npy:

- image.npy  : array (height, width, n_features) em reflectância/índices
- labels.npy : array (height, width) com class_id (0-4)
- meta.json  : metadados (caminho do GeoTIFF original, transform, crs)

Uso:
    python src/gee/build_unet_dataset.py                # defaults (Fernandópolis, Sentinel-2)
    python src/gee/build_unet_dataset.py --out data/processed --satellite sentinel2
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ee
import numpy as np
import rasterio
import requests

from src.gee.auth import initialize_gee
from src.gee.collector import get_composite_image, polygon_to_ee_geometry, search_images
from src.gee.ground_truth import get_mapbiomas_layer, reclassify_mapbiomas
from src.gee.preprocessor import preprocess_image
from src.processing.features import extract_features_ee, get_feature_bands

DEFAULT_POLYGON = [
    [-50.30, -20.25],
    [-50.20, -20.25],
    [-50.20, -20.30],
    [-50.30, -20.30],
    [-50.30, -20.25],
]

DEFAULT_BANDS = {
    "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12"],
    "landsat8": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
}


def download_array(ee_image: ee.Image, region: ee.Geometry, scale: int) -> np.ndarray:
    """Baixa uma imagem GEE (multi-banda) e retorna array (bandas, H, W).

    O GEE limita o tamanho da resposta de getDownloadURL (~50 MB). Para
    evitar estouro, as bandas são baixadas em grupos e concatenadas.
    Como todos os grupos usam a mesma região/escala/projeção, as grades
    ficam alinhadas.
    """
    bands = ee_image.bandNames().getInfo()
    arrays = []
    meta = None

    for i in range(0, len(bands), 6):
        group = bands[i : i + 6]
        url = ee_image.select(group).getDownloadURL(
            {"region": region, "scale": scale, "format": "GEO_TIFF", "maxPixels": 1e9}
        )

        response = requests.get(url, timeout=300)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            with rasterio.open(tmp_path) as src:
                chunk = src.read()
                if meta is None:
                    meta = {
                        "transform": src.transform,
                        "crs": str(src.crs),
                        "shape": chunk.shape,
                        "bounds": list(src.bounds),
                    }
        finally:
            os.unlink(tmp_path)

        arrays.append(chunk)

    data = np.concatenate(arrays, axis=0)
    return data, meta


def run(
    polygon,
    start_date,
    end_date,
    satellite,
    max_cloud_pct,
    scale,
    year,
    out_dir,
    image_name,
    labels_name,
):
    initialize_gee()

    geometry = polygon_to_ee_geometry(polygon)

    # 1. Imagem de satélite composta + pré-processamento + features
    collection = search_images(
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
        satellite=satellite,
        max_cloud_pct=max_cloud_pct,
    )

    n_images = collection.size().getInfo()
    print(f"Imagens encontradas: {n_images}")
    if n_images == 0:
        raise RuntimeError("Nenhuma imagem encontrada para os critérios selecionados.")

    composite = get_composite_image(collection, geometry)
    composite = preprocess_image(composite, satellite=satellite)
    features = extract_features_ee(composite, satellite=satellite)

    # Selecionar apenas as bandas de atributos (na ordem de get_feature_bands)
    band_names = get_feature_bands(satellite)
    feature_img = features.select(band_names)

    print(f"Bandas de atributos a baixar: {band_names}")

    image_data, meta = download_array(feature_img, geometry, scale)

    # 2. Ground truth MapBiomas reclassificado (0-4)
    gt_raw = get_mapbiomas_layer(year=year, geometry=geometry)
    gt_reclass = reclassify_mapbiomas(gt_raw)
    labels_data, _ = download_array(gt_reclass, geometry, scale)
    labels_data = labels_data[0].astype(np.int64)  # (H, W)

    # Alinhar dimensões de imagem e rótulo (mesma região/escala)
    h_img, w_img = image_data.shape[1], image_data.shape[2]
    f_h, f_w = labels_data.shape
    if (h_img, w_img) != (f_h, f_w):
        print(
            f"Aviso: dimensões diferentes imagem ({h_img}x{w_img}) vs "
            f"ground truth ({f_h}x{f_w}). Recortando para o mínimo."
        )
        min_h, min_w = min(h_img, f_h), min(w_img, f_w)
        image_data = image_data[:, :min_h, :min_w]
        labels_data = labels_data[:min_h, :min_w]

    # Transpor imagem para (H, W, C)
    image_chw = image_data.astype(np.float32)
    image_hwc = np.moveaxis(image_chw, 0, -1)

    os.makedirs(out_dir, exist_ok=True)

    image_path = os.path.join(out_dir, image_name)
    labels_path = os.path.join(out_dir, labels_name)
    meta_path = os.path.join(out_dir, "meta.json")

    np.save(image_path, image_hwc)
    np.save(labels_path, labels_data)

    meta_payload = {
        "satellite": satellite,
        "bands": band_names,
        "image_shape": list(image_hwc.shape),
        "labels_shape": list(labels_data.shape),
        "transform": str(meta["transform"]),
        "crs": meta["crs"],
        "bounds": meta["bounds"],
        "year": year,
        "scale": scale,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, ensure_ascii=False)

    print(f"Imagem salva em: {image_path}  {image_hwc.shape}  dtype={image_hwc.dtype}")
    print(f"Labels salvo em:  {labels_path} {labels_data.shape}  dtype={labels_data.dtype}")
    print(f"Metadados em:     {meta_path}")

    unique, counts = np.unique(labels_data, return_counts=True)
    print("\nDistribuição de classes no ground truth completo:")
    for cls_id, count in zip(unique, counts):
        print(f"  Classe {int(cls_id)}: {int(count)} pixels")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa imagem + ground truth completos para a U-Net"
    )
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--satellite", default="sentinel2", choices=["sentinel2", "landsat8"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--max-cloud", type=int, default=20)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--image-name", default="unet_image.npy")
    parser.add_argument("--labels-name", default="unet_labels.npy")
    args = parser.parse_args()

    run(
        polygon=DEFAULT_POLYGON,
        start_date=args.start,
        end_date=args.end,
        satellite=args.satellite,
        max_cloud_pct=args.max_cloud,
        scale=args.scale,
        year=args.year,
        out_dir=args.out,
        image_name=args.image_name,
        labels_name=args.labels_name,
    )


if __name__ == "__main__":
    main()
