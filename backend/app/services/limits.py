"""Limites de área e de download para a classificação via GEE.

O ``ee.Image.getDownloadURL`` aceita no máximo ~48 MB por requisição. Para
áreas grandes, o download é dividido em menos bandas por vez; áreas que nem
assim cabem são rejeitadas com uma mensagem clara.
"""

# Limite real do getDownloadURL (~48 MiB) com margem de segurança.
DOWNLOAD_BYTES_LIMIT = 48 * 1024 * 1024

# float32 = 4 bytes por pixel; 1.5 cobre o cabeçalho/compactação do GeoTIFF.
BYTES_PER_PIXEL = 4
SAFETY_FACTOR = 1.5


def estimated_pixels(area_ha: float, scale: int) -> int:
    """Estimativa de pixels de uma área (ha) na resolução informada."""
    if area_ha <= 0 or scale <= 0:
        return 0
    return int(area_ha * 10_000 / (scale * scale))


def area_limit_message(
    area_ha: float, scale: int, max_pixels: int
) -> str | None:
    """Retorna uma mensagem se a área excede o limite de pixels."""
    pixels = estimated_pixels(area_ha, scale)
    if pixels <= max_pixels:
        return None

    max_ha = max_pixels * (scale * scale) / 10_000
    return (
        f"Área selecionada (~{area_ha:,.0f} ha, {pixels:,} pixels) "
        f"excede o limite de {max_ha:,.0f} ha nesta resolução. "
        "Selecione uma área menor."
    )


def bands_per_download(estimated_pixels: int, max_bytes: int) -> int:
    """Quantas bandas cabem em uma requisição do getDownloadURL.

    Levanta ``RuntimeError`` quando nem uma banda cabe no limite.
    """
    bytes_per_band = estimated_pixels * BYTES_PER_PIXEL * SAFETY_FACTOR
    if bytes_per_band > max_bytes:
        raise RuntimeError(
            "Área muito grande para uma única classificação "
            f"(~{estimated_pixels:,} pixels). Selecione uma área menor."
        )
    return max(1, int(max_bytes // bytes_per_band))
