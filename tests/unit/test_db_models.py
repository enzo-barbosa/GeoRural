"""Testes do schema e dos repositórios de banco (SQLite em memória)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db import repositories as repo
from backend.app.db.base import Base


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)


def _geojson() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-50.30, -20.25],
                [-50.20, -20.25],
                [-50.20, -20.30],
                [-50.30, -20.30],
                [-50.30, -20.25],
            ]
        ],
    }


def test_create_user_normalizes_email(db):
    user = repo.create_user(
        db, email="  Teste@Example.com ", name="Teste", password_hash="hash"
    )

    assert user.id is not None
    assert user.email == "teste@example.com"
    assert user.is_active is True
    assert repo.get_user_by_email(db, "TESTE@example.com").id == user.id


def test_create_area_and_analysis_roundtrip(db):
    user = repo.create_user(db, email="a@b.com", name="Dono")
    area = repo.create_area(
        db,
        name="Fazenda",
        geojson=_geojson(),
        user_id=user.id,
        description="Área de teste",
        area_ha=12.5,
    )
    analysis = repo.create_analysis(
        db,
        job_id="job-123",
        satellite="sentinel2",
        classifier="auto",
        user_id=user.id,
        area_id=area.id,
        coverage="brazil",
        start_date="2023-01-01",
        end_date="2023-12-31",
        max_cloud=80,
        models_used=["brazil_rf", "brazil_svm"],
        weights={"brazil_rf": 0.5, "brazil_svm": 0.5},
        result={"stats": [{"class_name": "Pastagem", "pixels": 10}]},
    )

    assert analysis.id == "job-123"
    assert analysis.status == "queued"
    assert analysis.progress == 0
    assert analysis.result["stats"][0]["class_name"] == "Pastagem"
    assert area.geojson["type"] == "Polygon"
    assert repo.get_analysis(db, "job-123").area_id == area.id
    assert len(repo.list_areas(db, user_id=user.id)) == 1


def test_update_analysis_status(db):
    analysis = repo.create_analysis(
        db, job_id="job-status", satellite="sentinel2", classifier="svm"
    )

    repo.update_analysis_status(
        db, analysis, "completed", stage="done", progress=100
    )

    assert analysis.status == "completed"
    assert analysis.stage == "done"
    assert analysis.progress == 100
