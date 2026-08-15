"""Coleta de imagens do Google Earth Engine."""

import ee


def polygon_to_ee_geometry(coordinates: list) -> ee.Geometry:
    """Converte coordenadas de polígono para geometria GEE.

    Args:
        coordinates: Lista de coordenadas [[lon, lat], ...]

    Returns:
        Objeto ee.Geometry do polígono
    """
    return ee.Geometry.Polygon(coordinates)


def search_images(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    satellite: str = "sentinel2",
    max_cloud_pct: int = 20,
) -> ee.ImageCollection:
    """Busca imagens de satélite para uma área e período.

    Args:
        geometry: Geometria da área de interesse
        start_date: Data inicial (YYYY-MM-DD)
        end_date: Data final (YYYY-MM-DD)
        satellite: 'sentinel2' ou 'landsat8'
        max_cloud_pct: Percentual máximo de cobertura de nuvens

    Returns:
        Coleção de imagens filtrada
    """
    if satellite == "sentinel2":
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        )
    elif satellite == "landsat8":
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_pct))
        )
    else:
        raise ValueError(f"Satélite não suportado: {satellite}")

    return collection.sort("CLOUDY_PIXEL_PERCENTAGE")


def get_composite_image(
    collection: ee.ImageCollection, geometry: ee.Geometry
) -> ee.Image:
    """Gera mosaico mediano de uma coleção.

    O mosaico mediano combina múltiplas imagens, selecionando o valor
    mediano de cada pixel ao longo do tempo. Isso reduz efeitos de
    nuvens e ruídos.

    Args:
        collection: Coleção de imagens
        geometry: Geometria para recortar o resultado

    Returns:
        Imagem composta (mosaico mediano)
    """
    composite = collection.median().clip(geometry)
    return composite


def download_image(
    image: ee.Image,
    geometry: ee.Geometry,
    filepath: str,
    scale: int = 10,
    bandas: list | None = None,
) -> str:
    """Baixa uma imagem GEE para arquivo local.

    Para imagens grandes, usa exportação para Google Drive.
    Para imagens menores, baixa diretamente.

    Args:
        image: Imagem GEE
        geometry: Geometria da área
        filepath: Caminho para salvar o arquivo
        scale: Resolução em metros (10 para Sentinel-2, 30 para Landsat)
        bandas: Lista de bandas para baixar (None = todas)

    Returns:
        URL de download ou caminho do arquivo
    """
    if bandas:
        image = image.select(bandas)

    url = image.getDownloadURL(
        {"region": geometry, "scale": scale, "format": "GEO_TIFF", "maxPixels": 1e9}
    )

    return url


def export_to_drive(
    image: ee.Image,
    geometry: ee.Geometry,
    description: str,
    folder: str = "tcc-gee",
    scale: int = 10,
    crs: str = "EPSG:4326",
) -> ee.batch.Task:
    """Exporta imagem para Google Drive (para imagens grandes).

    Args:
        image: Imagem GEE
        geometry: Geometria da área
        description: Descrição da tarefa
        folder: Pasta no Google Drive
        scale: Resolução em metros
        crs: Sistema de coordenadas

    Returns:
        Objeto Task do GEE
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=geometry,
        scale=scale,
        maxPixels=1e10,
        crs=crs,
    )
    task.start()
    print(f"Exportação iniciada: {description}")
    print("Verifique em https://code.earthengine.google.com/tasks")
    return task
