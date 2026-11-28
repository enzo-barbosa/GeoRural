"""Schemas de áreas salvas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.schemas.classify import PolygonGeometry


class AreaCreate(BaseModel):
    """Dados para salvar uma área."""

    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    polygon: PolygonGeometry

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Informe um nome para a área.")
        return stripped


class AreaUpdate(BaseModel):
    """Campos editáveis de uma área."""

    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Informe um nome para a área.")
        return stripped


class AreaRead(BaseModel):
    """Área salva exposta pela API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None
    geojson: dict[str, Any]
    area_ha: float | None = None
    created_at: datetime
    updated_at: datetime
