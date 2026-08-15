"""Script de teste para download de imagem GEE."""

import sys

sys.path.insert(0, ".")

from src.gee.auth import initialize_gee
from src.gee.collector import get_composite_image, polygon_to_ee_geometry, search_images
from src.gee.preprocessor import preprocess_image
from src.processing.features import extract_features_ee

initialize_gee()

point_polygon = [
    [-50.30, -20.25],
    [-50.20, -20.25],
    [-50.20, -20.30],
    [-50.30, -20.30],
    [-50.30, -20.25],
]
geometry = polygon_to_ee_geometry(point_polygon)

collection = search_images(
    geometry=geometry,
    start_date="2024-01-01",
    end_date="2024-06-30",
    satellite="sentinel2",
    max_cloud_pct=15,
)

num_images = collection.size().getInfo()
print(f"Imagens encontradas: {num_images}")

if num_images == 0:
    print("Nenhuma imagem encontrada.")
    sys.exit(1)

composite = get_composite_image(collection, geometry)
composite = preprocess_image(composite, satellite="sentinel2")
features = extract_features_ee(composite, satellite="sentinel2")

print(f"Bandas disponíveis: {features.bandNames().getInfo()}")
print("Pipeline de dados executado com sucesso!")
