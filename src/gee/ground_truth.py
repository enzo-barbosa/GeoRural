"""Acesso aos rótulos de referência (MapBiomas e ESA WorldCover) via GEE.

Os crosswalks são explícitos: códigos não mapeados viram -1 e são mascarados,
evitando que o ``defaultValue=0`` do ``remap`` transforme água/mosaico em
"Vegetação Nativa".
"""

import ee

# ---------------------------------------------------------------------------
# Crosswalks (código de origem -> id interno)
# ---------------------------------------------------------------------------
# 0 Vegetação Nativa | 1 Pastagem | 2 Agricultura | 3 Solo Exposto
# 4 Área Construída | 5 Água
MAPBIOMAS_CROSSWALK: dict[int, int] = {
    1: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    9: 2,  # silvicultura
    10: 0,
    11: 0,  # área úmida -> vegetação
    12: 0,
    15: 1,
    18: 2,
    19: 2,
    20: 2,
    22: 3,
    23: 3,
    24: 4,
    25: 3,
    26: 5,
    29: 3,
    30: 3,
    31: 5,
    33: 5,
    35: 2,
    39: 2,
    40: 2,
    41: 2,
    46: 2,
    47: 2,
    48: 2,
}

# 0 Vegetação Nativa | 1 Pastagem/Vegetação Herbácea | 2 Agricultura
# 3 Solo Exposto | 4 Área Construída | 5 Água
WORLDCOVER_CROSSWALK: dict[int, int] = {
    10: 0,  # tree cover
    20: 0,  # shrubland
    30: 1,  # grassland (pasto manejado ou campo natural)
    40: 2,  # cropland
    50: 4,  # built-up
    60: 3,  # bare/sparse
    80: 5,  # water
    90: 0,  # wetland -> vegetação
    95: 0,  # mangrove
    70: -1,  # snow/ice -> ignorar
    100: -1,  # moss -> ignorar
}

# Cultivo: 0 Cana | 1 Citros | 2 Pastagem | 3 Soja | 4 Milho
MAPBIOMAS_CROP_CROSSWALK: dict[int, int] = {
    20: 0,
    47: 1,
    15: 2,
    39: 3,
    41: 4,  # aproximação: "outras lavouras temporárias" (inclui milho)
}


def _crosswalk_to_image(
    ground_truth: ee.Image, crosswalk: dict[int, int], band: str = "class_id"
) -> ee.Image:
    """Aplica um crosswalk com default -1 e mascara códigos ignorados."""
    source = ee.List(sorted(crosswalk.keys()))
    target = ee.List([crosswalk[code] for code in sorted(crosswalk.keys())])
    classified = ground_truth.remap(source, target, -1)
    classified = classified.updateMask(classified.gte(0))
    return classified.rename(band).toInt16()


def get_mapbiomas_layer(year: int = 2023, geometry: ee.Geometry = None) -> ee.Image:
    """Retorna a camada MapBiomas Collection 10 para o ano e região desejados."""
    mapbiomas = (
        ee.ImageCollection("projects/mapbiomas-public/assets/brazil/lulc/v1")
        .filter(ee.Filter.eq("collection_id", 10))
        .filter(ee.Filter.eq("year", year))
        .first()
    )

    ground_truth = mapbiomas.select("classification")

    if geometry is not None:
        ground_truth = ground_truth.clip(geometry)

    return ground_truth


def get_worldcover_layer(year: int = 2021, geometry: ee.Geometry = None) -> ee.Image:
    """Retorna a camada ESA WorldCover (v200 para 2021, v100 para 2020)."""
    asset = "ESA/WorldCover/v200" if year >= 2021 else "ESA/WorldCover/v100"
    ground_truth = ee.ImageCollection(asset).first().select("Map")

    if geometry is not None:
        ground_truth = ground_truth.clip(geometry)

    return ground_truth


def reclassify_mapbiomas(ground_truth: ee.Image) -> ee.Image:
    """Reclassifica MapBiomas para as 6 classes do projeto (0-5)."""
    return _crosswalk_to_image(ground_truth, MAPBIOMAS_CROSSWALK)


def reclassify_worldcover(ground_truth: ee.Image) -> ee.Image:
    """Reclassifica ESA WorldCover para as 6 classes do projeto (0-5)."""
    return _crosswalk_to_image(ground_truth, WORLDCOVER_CROSSWALK)


def reclassify_mapbiomas_crops(ground_truth: ee.Image) -> ee.Image:
    """Reclassifica MapBiomas para tipos de cultivo (0-4).

    O tipo de cultivo é Brasil-only; fora do Brasil a camada não é usada.
    """
    return _crosswalk_to_image(
        ground_truth, MAPBIOMAS_CROP_CROSSWALK, band="crop_id"
    )


def extract_samples_gee(
    image: ee.Image,
    ground_truth: ee.Image,
    region: ee.Geometry,
    scale: int = 30,
    seed: int = 42,
    samples_per_class: int = 500,
    class_band: str = "class_id",
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    """Extrai amostras de treinamento diretamente no GEE."""
    training_points = ground_truth.stratifiedSample(
        numPoints=samples_per_class,
        classBand=class_band,
        region=region,
        scale=scale,
        seed=seed,
        geometries=True,
    )

    training_data = image.sampleRegions(
        collection=training_points,
        scale=scale,
        geometries=True,
        tileScale=tile_scale,
    )

    def _attach_coords(feature):
        coords = feature.geometry().centroid(1).coordinates()
        return feature.set(
            {"longitude": coords.get(0), "latitude": coords.get(1)}
        )

    return training_data.map(_attach_coords)


def export_ground_truth(
    ground_truth: ee.Image, region: ee.Geometry, year: int, folder: str = "tcc_data"
):
    """Exporta o ground truth como GeoTIFF para o Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=ground_truth,
        description=f"ground_truth_{year}",
        folder=folder,
        region=region,
        scale=30,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print("Exportação iniciada. Verifique em https://code.earthengine.google.com/tasks")
