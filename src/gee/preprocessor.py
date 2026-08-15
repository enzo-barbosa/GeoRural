"""Pré-processamento de imagens orbitais."""

import ee


def mask_clouds_sentinel2(image: ee.Image) -> ee.Image:
    """Aplica máscara de nuvens em imagem Sentinel-2.

    Usa a banda SCL (Scene Classification Layer) para identificar
    nuvens, sombras de nuvens e pixels confiáveis.

    Args:
        image: Imagem Sentinel-2

    Returns:
        Imagem com máscara de nuvens aplicada
    """
    scl = image.select("SCL")

    mask = (
        scl.neq(4)
        .And(scl.neq(5))
        .And(scl.neq(6))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )

    return image.updateMask(mask)


def atmospheric_correction_sentinel2(image: ee.Image) -> ee.Image:
    """Aplica correção atmosférica na imagem Sentinel-2.

    Converte os valores digitais (DN) para reflectância de superfície
    usando os fatores de escala fornecidos pelo GEE.

    Fórmula: reflectância = DN * 0.0001 - 0.1

    Args:
        image: Imagem Sentinel-2 (nivel L2A - SR)

    Returns:
        Imagem com reflectância de superfície
    """
    optical = image.select("B.*").multiply(0.0001).add(-0.1)

    return image.addBands(optical, overwrite=True)


def mask_clouds_landsat8(image: ee.Image) -> ee.Image:
    """Aplica máscara de nuvens em imagem Landsat 8.

    Usa a banda QA_PIXEL para identificar nuvens e sombras.

    Args:
        image: Imagem Landsat 8

    Returns:
        Imagem com máscara de nuvens aplicada
    """
    qa = image.select("QA_PIXEL").toInt16()

    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)
    shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)

    mask = cloud_mask.And(shadow_mask)

    return image.updateMask(mask)


def atmospheric_correction_landsat8(image: ee.Image) -> ee.Image:
    """Aplica correção atmosférica na imagem Landsat 8.

    Converte os valores digitais para reflectância de superfície
    usando os fatores de escala do Collection 2.

    Fórmula: reflectância = DN * 0.0000275 - 0.2

    Args:
        image: Imagem Landsat 8 (Collection 2 Level 2)

    Returns:
        Imagem com reflectância de superfície
    """
    scale = 0.0000275
    offset = -0.2

    optical_bands = image.select("SR_B.").multiply(scale).add(offset)
    qa = image.select("QA_PIXEL")

    return image.addBands(optical_bands, overwrite=True).addBands(qa)


def preprocess_image(image: ee.Image, satellite: str = "sentinel2") -> ee.Image:
    """Aplica pré-processamento completo na imagem.

    Pipeline: correção atmosférica → máscara de nuvens

    Args:
        image: Imagem bruta do GEE
        satellite: 'sentinel2' ou 'landsat8'

    Returns:
        Imagem pré-processada
    """
    if satellite == "sentinel2":
        image = atmospheric_correction_sentinel2(image)
        image = mask_clouds_sentinel2(image)
    elif satellite == "landsat8":
        image = atmospheric_correction_landsat8(image)
        image = mask_clouds_landsat8(image)
    else:
        raise ValueError(f"Satélite não suportado: {satellite}")

    return image
