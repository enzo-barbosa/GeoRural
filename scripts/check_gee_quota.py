"""Teste rápido de cota do Google Earth Engine (1 tile pequeno).

Faz uma extração mínima (~2 km) pelo mesmo pipeline da U-Net
(Sentinel-2 -> mediana -> pré-processamento) e baixa um GeoTIFF pequeno.
Se passar, a conta tem cota para gerar os tiles da Etapa B.

Uso:
    python scripts/check_gee_quota.py
    python scripts/check_gee_quota.py --lon -48.55 --lat -22.49 --size-km 2
    python scripts/check_gee_quota.py --start 2023-01-01 --end 2023-12-31

Saída esperada:
    [ok] ... -> cota disponível (imprime tamanho do arquivo e tempo).
    [erro] ... -> mensagem do GEE e orientação.
"""

import argparse
import math
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import ee
import requests

from src.gee.auth import initialize_gee
from src.gee.collector import get_composite_image, search_images
from src.gee.preprocessor import preprocess_image


def _bbox(lon: float, lat: float, size_km: float) -> ee.Geometry:
    half_km = size_km / 2.0
    half_lat = half_km / 111.32
    half_lon = half_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return ee.Geometry.Rectangle(
        [lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat],
        geodesic=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Testa a cota do GEE")
    parser.add_argument("--lon", type=float, default=-48.55)
    parser.add_argument("--lat", type=float, default=-22.49)
    parser.add_argument("--size-km", type=float, default=2.0)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--max-cloud", type=int, default=20)
    parser.add_argument("--scale", type=int, default=10)
    args = parser.parse_args()

    started = time.time()
    try:
        initialize_gee()

        geometry = _bbox(args.lon, args.lat, args.size_km)
        print(
            f"Área de teste: {args.size_km} km em "
            f"({args.lon}, {args.lat}) — Sentinel-2 "
            f"{args.start}..{args.end}"
        )

        collection = search_images(
            geometry=geometry,
            start_date=args.start,
            end_date=args.end,
            satellite="sentinel2",
            max_cloud_pct=args.max_cloud,
        )
        n_images = collection.size().getInfo()
        print(f"Imagens encontradas: {n_images}")
        if n_images == 0:
            print("[erro] Nenhuma imagem no período. Ajuste --start/--end.")
            return 2

        composite = get_composite_image(collection, geometry)
        composite = preprocess_image(composite, satellite="sentinel2")
        composite = composite.select(["B2", "B3", "B4", "B8", "B11", "B12"])

        url = composite.getDownloadURL(
            {
                "scale": args.scale,
                "region": geometry,
                "format": "GEO_TIFF",
            }
        )
        response = requests.get(url, timeout=180)
        response.raise_for_status()

        elapsed = time.time() - started
        size_kb = len(response.content) / 1024.0
        print(f"[ok] download de {size_kb:.1f} KB em {elapsed:.1f} s")
        print("Cota disponível para gerar os tiles da Etapa B.")
        return 0

    except Exception as error:  # noqa: BLE001 - diagnóstico para o usuário
        elapsed = time.time() - started
        message = str(error)
        print(f"[erro] após {elapsed:.1f} s: {message[:800]}")
        lowered = message.lower()
        if "quota" in lowered or "too many requests" in lowered:
            print(
                "\nDiagnóstico: cota do GEE esgotada/restrita. "
                "Aguarde a renovação diária ou peça aumento de cota no "
                "Console do Earth Engine (projeto georural-project)."
            )
        elif "authentication" in lowered or "permission" in lowered:
            print(
                "\nDiagnóstico: problema de autenticação. Rode "
                "`earthengine authenticate` e confirme o projeto "
                "GEE_PROJECT_ID no .env."
            )
        elif "memory" in lowered or "computation" in lowered:
            print(
                "\nDiagnóstico: limite de processamento. Tente uma área "
                "menor (--size-km 1) para separar cota de complexidade."
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
