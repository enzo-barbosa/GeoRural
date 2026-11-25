"""Schemas de histórico e detalhe de análises."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AnalysisSummary(BaseModel):
    """Análise listada no histórico (sem o payload completo)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str | None = None
    status: str
    stage: str | None = None
    progress: int
    error: str | None = None
    satellite: str
    classifier: str
    coverage: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    area_id: int | None = None
    area_name: str | None = None
    area_ha: float | None = None
    num_images: int | None = None
    confidence_mean: float | None = None
    image_url: str | None = None
    created_at: datetime
    updated_at: datetime


class AnalysisRead(AnalysisSummary):
    """Análise completa, incluindo o resultado da classificação."""

    label_source: str | None = None
    max_cloud: int | None = None
    include_topography: bool
    include_crop: bool
    postprocess: bool
    models_used: list[str] | None = None
    weights: dict[str, float] | None = None
    geojson: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
