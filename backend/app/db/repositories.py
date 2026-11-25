"""Operações de banco usadas pelos próximos PRs (auth e persistência)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

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


def get_area(db: Session, area_id: int, user_id: int | None = None) -> Area | None:
    """Busca uma área, opcionalmente exigindo que pertença ao usuário."""
    area = db.get(Area, area_id)
    if area is None:
        return None
    if user_id is not None and area.user_id != user_id:
        return None
    return area


def update_area(
    db: Session,
    area: Area,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Area:
    """Atualiza os campos editáveis de uma área."""
    if name is not None:
        area.name = name.strip()
    if description is not None:
        stripped = description.strip()
        area.description = stripped or None
    db.commit()
    db.refresh(area)
    return area


def delete_area(db: Session, area: Area) -> None:
    """Remove uma área; as análises relacionadas ficam com area_id nulo."""
    db.delete(area)
    db.commit()


def list_analyses(
    db: Session, user_id: int, *, limit: int = 20, offset: int = 0
) -> list[Analysis]:
    """Lista as análises do usuário, mais recentes primeiro."""
    query = (
        select(Analysis)
        .options(selectinload(Analysis.area))
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(query).all())


def get_analysis_for_user(
    db: Session, job_id: str, user_id: int
) -> Analysis | None:
    """Busca uma análise garantindo que pertence ao usuário."""
    query = (
        select(Analysis)
        .options(selectinload(Analysis.area))
        .where(Analysis.id == job_id, Analysis.user_id == user_id)
    )
    return db.scalar(query)


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
