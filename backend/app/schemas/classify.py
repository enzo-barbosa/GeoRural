"""Pydantic schemas for classification API."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class PolygonGeometry(BaseModel):
    """GeoJSON Polygon geometry."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]]  # [[[lon, lat], ...]]


class ClassifyRequest(BaseModel):
    """Request to classify an area."""

    polygon: PolygonGeometry
    satellite: Literal["sentinel2", "landsat8"]
    classifier: Literal["random_forest", "svm", "unet_cnn"]
    start_date: date
    end_date: date
    max_cloud: int = Field(default=80, ge=0, le=100)


class ClassStats(BaseModel):
    """Statistics for a single class."""

    class_name: str
    pixels: int
    percent: float


class ClassificationResult(BaseModel):
    """Successful classification result."""

    job_id: str
    status: Literal["completed"] = "completed"
    satellite: str
    classifier: str
    start_date: str
    end_date: str
    num_images: int
    stats: list[ClassStats]
    classification_image_url: str
    rgb_image_url: str | None = None
    overlay_image_url: str | None = None


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

    random_forest: bool
    svm: bool
    unet_cnn: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"] = "healthy"
    gee_initialized: bool