"""Acesso ao MapBiomas Collection 10 via Google Earth Engine."""

import ee


def get_mapbiomas_layer(year: int = 2023, geometry: ee.Geometry = None) -> ee.Image:
    """Retorna a camada MapBiomas Collection 10 para o ano e região desejados.

    Args:
        year: Ano da classificação (1985-2023 para Collection 10)
        geometry: Região de interesse (recortará o resultado)

    Returns:
        Imagem GEE com a camada de cobertura do solo do MapBiomas
    """
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


def reclassify_mapbiomas(ground_truth: ee.Image) -> ee.Image:
    """Reclassifica os códigos do MapBiomas para as 5 classes do projeto.

    Mapeamento:
    - 0: Vegetação Nativa
    - 1: Pastagem
    - 2: Agricultura
    - 3: Solo Exposto
    - 4: Área Construída

    Args:
        ground_truth: Imagem MapBiomas com banda 'classification'

    Returns:
        Imagem com banda 'class_id' (0-4)
    """
    class_mapping = [
        [1, 0],  # Vegetação Nativa
        [3, 0],
        [4, 0],
        [5, 0],
        [6, 0],
        [10, 0],
        [11, 0],
        [12, 0],
        [15, 1],  # Pastagem
        [18, 2],  # Agricultura
        [19, 2],
        [20, 2],
        [22, 3],  # Solo Exposto
        [23, 3],
        [24, 4],  # Área Construída
        [25, 3],
        [29, 3],
        [30, 3],
        [39, 2],
        [40, 2],
        [41, 2],
        [46, 2],
        [47, 2],
        [48, 2],
    ]

    classified = ground_truth.remap(
        from_=ee.List([x[0] for x in class_mapping]),
        to=ee.List([x[1] for x in class_mapping]),
    )

    return classified.rename("class_id").toByte()


def extract_samples_gee(
    image: ee.Image,
    ground_truth: ee.Image,
    region: ee.Geometry,
    scale: int = 30,
    seed: int = 42,
    samples_per_class: int = 500,
) -> ee.FeatureCollection:
    """Extrai amostras de treinamento diretamente no GEE.

    Args:
        image: Imagem com todas as bandas de atributos
        ground_truth: Imagem de ground truth reclassificada (0-4)
        region: Região de interesse
        scale: Resolução em metros
        seed: Seed para amostragem
        samples_per_class: Número de amostras por classe

    Returns:
        FeatureCollection com pontos e valores de bandas
    """
    training_points = ground_truth.stratifiedSample(
        numPoints=samples_per_class,
        classBand="class_id",
        region=region,
        scale=scale,
        seed=seed,
        geometries=True,
    )

    training_data = image.sampleRegions(
        collection=training_points, scale=scale, geometries=True
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
    """Exporta o ground truth como GeoTIFF para o Google Drive.

    Args:
        ground_truth: Imagem reclassificada (0-4)
        region: Região de exportação
        year: Ano do MapBiomas
        folder: Pasta no Google Drive
    """
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
