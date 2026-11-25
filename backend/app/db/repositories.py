"""Operações de banco usadas pelos próximos PRs (auth e persistência)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Analysis, Area, User


def create_user(
    db: Session,
    *,
    email: str,
    name: str,
    password_hash: str | None = None,
) -> User:
    """Cria um usuário com email normalizado (minúsculo)."""
    user = User(
        email=email.strip().lower(),
        name=name.strip(),
        password_hash=password_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    """Busca usuário por email (case-insensitive)."""
    return db.scalar(
        select(User).where(User.email == email.strip().lower())
    )


def create_area(
    db: Session,
    *,
    name: str,
    geojson: dict[str, Any],
    user_id: int | None = None,
    description: str | None = None,
    area_ha: float | None = None,
) -> Area:
    """Cria uma área salva."""
    area = Area(
        user_id=user_id,
        name=name.strip(),
        description=description,
        geojson=geojson,
        area_ha=area_ha,
    )
    db.add(area)
    db.commit()
    db.refresh(area)
    return area


def list_areas(db: Session, user_id: int | None = None) -> list[Area]:
    """Lista áreas, opcionalmente filtrando por usuário."""
    query = select(Area).order_by(Area.created_at.desc())
    if user_id is not None:
        query = query.where(Area.user_id == user_id)
    return list(db.scalars(query).all())


def create_analysis(
    db: Session,
    *,
    job_id: str,
    satellite: str,
    classifier: str,
    user_id: int | None = None,
    area_id: int | None = None,
    **fields: Any,
) -> Analysis:
    """Cria uma análise (job) no banco."""
    analysis = Analysis(
        id=job_id,
        user_id=user_id,
        area_id=area_id,
        satellite=satellite,
        classifier=classifier,
        **fields,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def get_analysis(db: Session, job_id: str) -> Analysis | None:
    """Busca análise pelo job_id."""
    return db.get(Analysis, job_id)


def update_analysis_status(
    db: Session,
    analysis: Analysis,
    status: str,
    *,
    stage: str | None = None,
    progress: int | None = None,
    error: str | None = None,
) -> Analysis:
    """Atualiza o estado de processamento de uma análise."""
    analysis.status = status
    if stage is not None:
        analysis.stage = stage
    if progress is not None:
        analysis.progress = progress
    if error is not None:
        analysis.error = error
    db.commit()
    db.refresh(analysis)
    return analysis
