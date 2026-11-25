"""Modelos de banco de dados do GeoRural."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


def _utcnow() -> datetime:
    """Data/hora atual em UTC (portável entre Postgres e SQLite)."""
    return datetime.now(UTC)


class User(Base):
    """Usuário da aplicação (o hash de senha é preenchido no fluxo de auth)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    areas: Mapped[list[Area]] = relationship(back_populates="user")
    analyses: Mapped[list[Analysis]] = relationship(back_populates="user")


class Area(Base):
    """Área salva pelo usuário (polígono GeoJSON + metadados)."""

    __tablename__ = "areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    geojson: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    area_ha: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    user: Mapped[User | None] = relationship(back_populates="areas")
    analyses: Mapped[list[Analysis]] = relationship(back_populates="area")


class Analysis(Base):
    """Execução de classificação e seu resultado persistido."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    area_id: Mapped[int | None] = mapped_column(
        ForeignKey("areas.id", ondelete="SET NULL"), index=True, nullable=True
    )
    name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    geojson: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    area_ha: Mapped[float | None] = mapped_column(Float, nullable=True)

    satellite: Mapped[str] = mapped_column(String(20), nullable=False)
    classifier: Mapped[str] = mapped_column(String(40), nullable=False)
    coverage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    label_source: Mapped[str | None] = mapped_column(String(80), nullable=True)
    start_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    max_cloud: Mapped[int | None] = mapped_column(Integer, nullable=True)
    include_topography: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    include_crop: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    postprocess: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(20), default="queued", index=True, nullable=False
    )
    stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    num_images: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence_mean: Mapped[float | None] = mapped_column(Float, nullable=True)

    models_used: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    weights: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    user: Mapped[User | None] = relationship(back_populates="analyses")
    area: Mapped[Area | None] = relationship(back_populates="analyses")
