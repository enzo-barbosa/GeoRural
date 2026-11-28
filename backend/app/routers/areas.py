"""Rotas de áreas salvas pelo usuário."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.models import Area, User
from backend.app.db.repositories import (
    create_area,
    delete_area,
    get_area,
    list_areas,
    update_area,
)
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.schemas.area import AreaCreate, AreaRead, AreaUpdate
from backend.app.services.geometry_validation import (
    GeometryValidationError,
    ValidatedPolygon,
    validate_polygon,
)

router = APIRouter(prefix="/api/areas", tags=["areas"])

_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Área não encontrada"
)


def _validated_polygon(payload: AreaCreate) -> ValidatedPolygon:
    """Valida a geometria e devolve o GeoJSON normalizado com as medidas."""
    try:
        return validate_polygon(payload.polygon.model_dump())
    except GeometryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[AreaRead])
def list_user_areas(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Area]:
    """Lista as áreas salvas do usuário autenticado."""
    return list_areas(db, user_id=current_user.id)


@router.post("", response_model=AreaRead, status_code=201)
def create_user_area(
    payload: AreaCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Area:
    """Salva um novo polígono como área do usuário."""
    validated = _validated_polygon(payload)
    return create_area(
        db,
        name=payload.name,
        description=payload.description,
        geojson=validated.geojson,
        user_id=current_user.id,
        area_ha=round(validated.area_ha, 4),
    )


@router.get("/{area_id}", response_model=AreaRead)
def get_user_area(
    area_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Area:
    """Detalhe de uma área do usuário."""
    area = get_area(db, area_id, user_id=current_user.id)
    if area is None:
        raise _NOT_FOUND
    return area


@router.patch("/{area_id}", response_model=AreaRead)
def rename_user_area(
    area_id: int,
    payload: AreaUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Area:
    """Renomeia ou atualiza a descrição de uma área."""
    area = get_area(db, area_id, user_id=current_user.id)
    if area is None:
        raise _NOT_FOUND
    return update_area(
        db, area, name=payload.name, description=payload.description
    )


@router.delete("/{area_id}", status_code=204)
def delete_user_area(
    area_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Remove uma área do usuário (o histórico de análises é preservado)."""
    area = get_area(db, area_id, user_id=current_user.id)
    if area is None:
        raise _NOT_FOUND
    delete_area(db, area)
