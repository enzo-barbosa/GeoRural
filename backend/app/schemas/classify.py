"""Pydantic schemas for classification API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PolygonGeometry(BaseModel):
    """GeoJSON Polygon geometry."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]]  # [[[lon, lat], ...]]


class ClassifyRequest(BaseModel):
    """Request to classify an area."""

    polygon: PolygonGeometry
    satellite: Literal["sentinel2", "landsat8"]
    classifier: Literal[
        "auto",
        "ensemble",
        "random_forest",
        "svm",
        "unet_cnn",
        "brazil_rf",
        "brazil_svm",
        "brazil_unet",
        "global_rf",
        "global_svm",
    ] = "auto"
    start_date: date
    end_date: date
    max_cloud: int = Field(default=80, ge=0, le=100)
    include_topography: bool = True
    include_crop: bool = True


class ClassStats(BaseModel):
    """Statistics for a single class."""

    class_name: str
    pixels: int
    percent: float


class MeasurementStats(BaseModel):
    """Área/perímetro reais da área analisada."""

    total_area_ha: float
    perimeter_m: float
    total_pixels: int
    parcels: int


class ClassArea(BaseModel):
    """Área real (ha) de uma classe/talhões."""

    class_name: str
    area_ha: float
    pixels: int
    percent: float
    parcels: int


class SlopeClassArea(BaseModel):
    """Área por classe de declividade."""

    class_name: str
    area_ha: float
    percent: float


class TopographyStats(BaseModel):
    """Análise topográfica do terreno."""

    elevation_min: float | None = None
    elevation_max: float | None = None
    elevation_mean: float | None = None
    relief_m: float | None = None
    slope_min: float | None = None
    slope_mean: float | None = None
    slope_max: float | None = None
    dominant_class: str | None = None
    classes: list[SlopeClassArea] = []


class LayerLegendItem(BaseModel):
    """Item de legenda de uma camada."""

    label: str
    color: str


class LayerInfo(BaseModel):
    """Camada geoespacial para sobreposição no mapa."""

    name: str
    label: str
    image_url: str
    legend: list[LayerLegendItem] = []


class ClassificationResult(BaseModel):
    """Successful classification result."""

    job_id: str
    status: Literal["completed"] = "completed"
    satellite: str
    classifier: str
    coverage: Literal["brazil", "global"] = "brazil"
    label_source: str | None = None
    models_used: list[str] = []
    weights: dict[str, float] = {}
    confidence_mean: float | None = None
    start_date: str
    end_date: str
    num_images: int
    stats: list[ClassStats]
    measurements: MeasurementStats | None = None
    class_areas: list[ClassArea] = []
    topography: TopographyStats | None = None
    crop_stats: list[ClassArea] | None = None
    layers: list[LayerInfo] = []
    classification_image_url: str
    rgb_image_url: str | None = None
    overlay_image_url: str | None = None
    bounds: list[list[float]] | None = None


class ClassificationError(BaseModel):
    """Failed classification result."""

    job_id: str
    status: Literal["failed"] = "failed"
    error: str


class ClassificationPending(BaseModel):
    """Pending classification result."""

    job_id: str
    status: Literal["pending", "processing"] = "pending"


ClassifyResponse = ClassificationResult | ClassificationError | ClassificationPending


class ModelsStatusResponse(BaseModel):
    """Model availability status."""

    brazil_rf: bool = False
    brazil_svm: bool = False
    brazil_unet: bool = False
    global_rf: bool = False
    global_svm: bool = False
    crop: bool = False


class CoverageRequest(BaseModel):
    """Polígono para consulta de cobertura."""

    polygon: PolygonGeometry


class CoverageResponse(BaseModel):
    """Cobertura (Brasil/global) de um polígono."""

    coverage: Literal["brazil", "global"]
    label_source: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"] = "healthy"
    gee_initialized: bool
