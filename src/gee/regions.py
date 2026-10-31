"""Regiões de amostragem para os domínios Brasil e global.

As caixas são representativas (não exaustivas) e foram escolhidas para cobrir
biomas e sistemas agrícolas distintos, minimizando sobreposição.
"""

from dataclasses import dataclass

import ee

BRAZIL_YEARS = [2019, 2020, 2021, 2022, 2023]
GLOBAL_YEAR = 2021


@dataclass(frozen=True)
class Region:
    """Região de amostragem com caixa geográfica (lon/lat)."""

    name: str
    domain: str
    biome: str
    country: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat

    def to_ee_geometry(self) -> ee.Geometry:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ee.Geometry.Rectangle(
            [min_lon, min_lat, max_lon, max_lat],
            proj="EPSG:4326",
            geodesic=False,
        )

    def bounds(self) -> tuple[float, float, float, float]:
        return self.bbox


BRAZIL_REGIONS: list[Region] = [
    Region(
        "amazonia", "brazil", "Amazônia", "Brazil",
        (-62.0, -6.0, -59.0, -3.0),
    ),
    Region(
        "cerrado", "brazil", "Cerrado", "Brazil",
        (-49.0, -14.0, -46.0, -11.0),
    ),
    Region(
        "mata_atlantica", "brazil", "Mata Atlântica", "Brazil",
        (-45.0, -22.0, -42.0, -19.0),
    ),
    Region(
        "caatinga", "brazil", "Caatinga", "Brazil",
        (-41.0, -11.0, -38.0, -8.0),
    ),
    Region(
        "pampa", "brazil", "Pampa", "Brazil",
        (-55.0, -32.0, -52.0, -30.0),
    ),
    Region(
        "pantanal", "brazil", "Pantanal", "Brazil",
        (-57.0, -19.0, -55.0, -17.0),
    ),
    # Polos agrícolas para o classificador de cultivo.
    Region(
        "piracicaba", "brazil", "Mata Atlântica", "Brazil",
        (-47.8, -22.9, -47.4, -22.5),
    ),
    Region(
        "ribeirao_preto", "brazil", "Cerrado", "Brazil",
        (-47.9, -21.4, -47.5, -21.0),
    ),
    Region(
        "barretos", "brazil", "Cerrado", "Brazil",
        (-48.8, -20.7, -48.4, -20.3),
    ),
    Region(
        "rio_verde", "brazil", "Cerrado", "Brazil",
        (-51.0, -17.9, -50.6, -17.5),
    ),
    Region(
        "barreiras", "brazil", "Cerrado", "Brazil",
        (-45.2, -12.3, -44.8, -11.9),
    ),
    Region(
        "sorriso", "brazil", "Cerrado", "Brazil",
        (-55.8, -12.7, -55.4, -12.3),
    ),
]

# Tiles pequenos (~6,7 km) para a U-Net, que precisa de rasters completos.
UNET_REGIONS: list[Region] = [
    Region(
        "unet_amazonia", "brazil", "Amazônia", "Brazil",
        (-62.0, -6.0, -61.94, -5.94),
    ),
    Region(
        "unet_cerrado", "brazil", "Cerrado", "Brazil",
        (-49.0, -14.0, -48.94, -13.94),
    ),
    Region(
        "unet_mata_atlantica", "brazil", "Mata Atlântica", "Brazil",
        (-45.0, -22.0, -44.94, -21.94),
    ),
    Region(
        "unet_caatinga", "brazil", "Caatinga", "Brazil",
        (-41.0, -11.0, -40.94, -10.94),
    ),
    Region(
        "unet_pampa", "brazil", "Pampa", "Brazil",
        (-55.0, -32.0, -54.94, -31.94),
    ),
    Region(
        "unet_pantanal", "brazil", "Pantanal", "Brazil",
        (-57.0, -19.0, -56.94, -18.94),
    ),
    Region(
        "unet_piracicaba", "brazil", "Mata Atlântica", "Brazil",
        (-47.8, -22.9, -47.74, -22.84),
    ),
    Region(
        "unet_rio_verde", "brazil", "Cerrado", "Brazil",
        (-51.0, -17.9, -50.94, -17.84),
    ),
]

GLOBAL_REGIONS: list[Region] = [
    Region(
        "us_corn_belt", "global", "Corn Belt", "United States",
        (-93.5, 41.0, -91.5, 42.5),
    ),
    Region(
        "us_great_plains", "global", "Great Plains", "United States",
        (-101.0, 38.0, -99.0, 40.0),
    ),
    Region(
        "us_central_valley", "global", "Central Valley", "United States",
        (-121.5, 36.5, -119.5, 38.5),
    ),
    Region(
        "pampas", "global", "Pampas", "Argentina",
        (-61.0, -35.0, -59.0, -33.0),
    ),
    Region(
        "amazonia_peru", "global", "Amazônia", "Peru",
        (-73.5, -8.0, -71.5, -6.0),
    ),
    Region(
        "andes", "global", "Andes", "Bolivia",
        (-70.0, -16.0, -68.0, -14.0),
    ),
    Region(
        "europa_ocidental", "global", "Europa Ocidental", "France",
        (1.0, 47.0, 3.0, 49.0),
    ),
    Region(
        "mediterraneo", "global", "Mediterrâneo", "Spain",
        (-3.0, 38.0, -1.0, 40.0),
    ),
    Region(
        "sahel", "global", "Sahel", "Mali",
        (-2.0, 14.0, 0.0, 16.0),
    ),
    Region(
        "africa_oriental", "global", "Savana da África Oriental", "Kenya",
        (36.0, -2.0, 38.0, 0.0),
    ),
    Region(
        "africa_austral", "global", "África Austral", "South Africa",
        (25.0, -27.0, 27.0, -25.0),
    ),
    Region(
        "india", "global", "Punjab/Ganges", "India",
        (75.0, 29.0, 77.0, 31.0),
    ),
    Region(
        "sudeste_asiatico", "global", "Sudeste Asiático", "Thailand",
        (100.0, 14.0, 102.0, 16.0),
    ),
    Region(
        "china", "global", "Planície da China", "China",
        (115.0, 35.0, 117.0, 37.0),
    ),
    Region(
        "australia", "global", "Murray-Darling", "Australia",
        (145.0, -35.0, 147.0, -33.0),
    ),
]


def list_regions(domain: str) -> list[Region]:
    """Lista regiões de um domínio ('brazil' ou 'global')."""
    if domain == "brazil":
        return BRAZIL_REGIONS
    if domain == "global":
        return GLOBAL_REGIONS
    raise ValueError(f"Domínio inválido: {domain}")


def get_region(name: str) -> Region:
    """Busca uma região pelo nome em todos os conjuntos."""
    for region in BRAZIL_REGIONS + UNET_REGIONS + GLOBAL_REGIONS:
        if region.name == name:
            return region
    raise KeyError(f"Região desconhecida: {name}")


def resolve_unet_regions(names: list[str] | None) -> list[Region]:
    """Resolve 'all' ou nomes de tiles pequenos para a U-Net."""
    if not names or names == ["all"]:
        return UNET_REGIONS

    selected = []
    for name in names:
        region = get_region(name)
        if region.name not in {r.name for r in UNET_REGIONS}:
            raise ValueError(
                f"{name} não é um tile de U-Net. Use: "
                f"{[r.name for r in UNET_REGIONS]}"
            )
        selected.append(region)
    return selected


def resolve_regions(domain: str, names: list[str] | None) -> list[Region]:
    """Resolve 'all' ou uma lista de nomes para objetos Region."""
    regions = list_regions(domain)
    if not names or names == ["all"]:
        return regions

    selected = []
    for name in names:
        region = get_region(name)
        if region.domain != domain:
            raise ValueError(
                f"Região {name} pertence ao domínio {region.domain}, não {domain}."
            )
        selected.append(region)
    return selected
