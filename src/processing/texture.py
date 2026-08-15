"""Extração de textura GLCM via Google Earth Engine."""

import ee


def calculate_glcm_ee(
    image: ee.Image, band_name: str = "B4", kernel_size: int = 3
) -> ee.Image:
    """Calcula texturas GLCM diretamente no GEE.

    Usa a função glcmTexture() do GEE que aplica uma janela deslizante
    (kernel) sobre a imagem, calculando as propriedades de textura
    pixel a pixel.

    Args:
        image: Imagem GEE
        band_name: Nome da banda para calcular GLCM
        kernel_size: Tamanho do kernel (janela)

    Returns:
        Imagem com bandas de textura adicionadas (contrast, dissimilarity)
    """
    band = image.select(band_name)
    band_gray = band.multiply(255).toUint8()

    glcm = band_gray.glcmTexture(size=kernel_size)

    contrast = glcm.select(f"{band_name}_contrast").rename("contrast")
    dissimilarity = glcm.select(f"{band_name}_diss").rename("dissimilarity")

    return image.addBands([contrast, dissimilarity])
