"""Testes de áreas, histórico, persistência da classificação e outputs."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import get_settings
from backend.app.db.base import Base
from backend.app.db.models import Analysis
from backend.app.db.session import get_db
from backend.app.dependencies.models import get_gee
from backend.app.main import create_app

POLYGON = {
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


@pytest.fixture()
def client(db):
    app = create_app()

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_gee] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _register(client, email="user@example.com", name="Usuário"):
    response = client.post(
        "/api/auth/register",
        json={"email": email, "name": name, "password": "senha1234"},
    )
    assert response.status_code == 201
    return response.json()


def _create_area(client, name="Fazenda"):
    response = client.post(
        "/api/areas",
        json={"name": name, "description": "Área de teste", "polygon": POLYGON},
    )
    assert response.status_code == 201, response.text
    return response.json()


# ------------------------------- áreas -------------------------------------


def test_area_crud(client):
    _register(client)
    created = _create_area(client)

    assert created["id"] > 0
    assert created["name"] == "Fazenda"
    assert created["geojson"]["type"] == "Polygon"
    assert created["area_ha"] > 0

    listed = client.get("/api/areas")
    assert listed.status_code == 200
    assert [a["id"] for a in listed.json()] == [created["id"]]

    detail = client.get(f"/api/areas/{created['id']}")
    assert detail.status_code == 200
    assert detail.json()["name"] == "Fazenda"

    renamed = client.patch(
        f"/api/areas/{created['id']}", json={"name": "Fazenda Nova"}
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Fazenda Nova"

    deleted = client.delete(f"/api/areas/{created['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/areas/{created['id']}").status_code == 404


def test_areas_are_isolated_between_users(client):
    _register(client, email="a@example.com", name="A")
    area = _create_area(client)
    client.post("/api/auth/logout")

    _register(client, email="b@example.com", name="B")

    assert client.get("/api/areas").json() == []
    assert client.get(f"/api/areas/{area['id']}").status_code == 404
    assert (
        client.patch(
            f"/api/areas/{area['id']}", json={"name": "Invasão"}
        ).status_code
        == 404
    )
    assert client.delete(f"/api/areas/{area['id']}").status_code == 404


def test_area_requires_authentication(client):
    assert client.get("/api/areas").status_code == 401
    assert client.post("/api/areas", json={"name": "X", "polygon": POLYGON}).status_code == 401


# ------------------------------ análises -----------------------------------


def _seed_analysis(db, user_id, job_id="job-1", **overrides):
    analysis = Analysis(
        id=job_id,
        user_id=user_id,
        satellite="sentinel2",
        classifier="auto",
        status="completed",
        progress=100,
        result={"classification_image_url": f"/outputs/{job_id}/classification.png"},
    )
    for key, value in overrides.items():
        setattr(analysis, key, value)
    db.add(analysis)
    db.commit()
    return analysis


def test_analysis_history_and_detail(client, db):
    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-1", name="Primeira")

    listed = client.get("/api/analyses")
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 1
    assert body[0]["id"] == "job-1"
    assert body[0]["name"] == "Primeira"
    assert body[0]["image_url"] == "/outputs/job-1/classification.png"

    detail = client.get("/api/analyses/job-1")
    assert detail.status_code == 200
    assert detail.json()["result"]["classification_image_url"].endswith(".png")

    assert client.get("/api/analyses/inexistente").status_code == 404


def test_analysis_history_is_isolated(client, db):
    first = _register(client, email="a@example.com", name="A")
    _seed_analysis(db, first["id"], job_id="job-a")
    client.post("/api/auth/logout")

    _register(client, email="b@example.com", name="B")

    assert client.get("/api/analyses").json() == []
    assert client.get("/api/analyses/job-a").status_code == 404


# ------------------------------- outputs -----------------------------------


def test_outputs_are_protected_by_owner(client, db, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-out")

    output_dir = Path(tmp_path) / "job-out"
    output_dir.mkdir(parents=True)
    (output_dir / "classification.png").write_bytes(b"png")

    assert (
        client.get("/outputs/job-out/classification.png").status_code == 200
    )

    client.post("/api/auth/logout")
    assert client.get("/outputs/job-out/classification.png").status_code == 401

    _register(client, email="outro@example.com", name="Outro")
    assert (
        client.get("/outputs/job-out/classification.png").status_code == 404
    )

    assert client.get("/outputs/job-out/segredo.txt").status_code == 404
    assert (
        client.get("/outputs/job-out/..%2F..%2Fetc%2Fpasswd.png").status_code
        == 404
    )


# --------------------- persistência da classificação -----------------------


def _fake_result(job_id: str, status: str = "completed") -> dict:
    if status == "failed":
        return {"job_id": job_id, "status": "failed", "error": "sem imagens"}
    return {
        "job_id": job_id,
        "status": "completed",
        "satellite": "sentinel2",
        "classifier": "auto",
        "coverage": "brazil",
        "label_source": "MapBiomas",
        "models_used": ["brazil_rf"],
        "weights": {"brazil_rf": 1.0},
        "confidence_mean": 0.9,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "num_images": 3,
        "stats": [
            {"class_name": "Pastagem", "pixels": 10, "percent": 100.0}
        ],
        "measurements": {
            "total_area_ha": 12.5,
            "perimeter_m": 1500.0,
            "total_pixels": 10,
            "parcels": 1,
        },
        "class_areas": [
            {
                "class_name": "Pastagem",
                "area_ha": 12.5,
                "pixels": 10,
                "percent": 100.0,
                "parcels": 1,
            }
        ],
        "topography": None,
        "crop_stats": None,
        "postprocessed": True,
        "layers": [],
        "classification_image_url": f"/outputs/{job_id}/classification.png",
        "rgb_image_url": None,
        "overlay_image_url": None,
        "bounds": None,
    }


def test_classification_is_persisted(client, db, monkeypatch):
    import backend.app.routers.classify as classify_router

    async def fake_classify(**kwargs):
        return _fake_result(kwargs["job_id"])

    monkeypatch.setattr(classify_router, "classify_area", fake_classify)

    user = _register(client)
    area = _create_area(client)

    response = client.post(
        "/api/classify",
        json={
            "polygon": POLYGON,
            "satellite": "sentinel2",
            "classifier": "auto",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "area_id": area["id"],
            "name": "Safra 2023",
        },
    )

    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    analysis = db.get(Analysis, job_id)
    assert analysis is not None
    assert analysis.user_id == user["id"]
    assert analysis.area_id == area["id"]
    assert analysis.name == "Safra 2023"
    assert analysis.status == "completed"
    assert analysis.progress == 100
    assert analysis.result["measurements"]["total_area_ha"] == 12.5

    detail = client.get(f"/api/analyses/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["area_name"] == "Fazenda"


def test_failed_classification_is_persisted(client, db, monkeypatch):
    import backend.app.routers.classify as classify_router

    async def fake_classify(**kwargs):
        return _fake_result(kwargs["job_id"], status="failed")

    monkeypatch.setattr(classify_router, "classify_area", fake_classify)

    _register(client)
    response = client.post(
        "/api/classify",
        json={
            "polygon": POLYGON,
            "satellite": "sentinel2",
            "classifier": "auto",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"

    analysis = db.get(Analysis, body["job_id"])
    assert analysis is not None
    assert analysis.status == "failed"
    assert analysis.error == "sem imagens"


def test_classification_rejects_foreign_area(client, db, monkeypatch):
    import backend.app.routers.classify as classify_router

    async def fake_classify(**kwargs):  # pragma: no cover - não deve rodar
        return _fake_result(kwargs["job_id"])

    monkeypatch.setattr(classify_router, "classify_area", fake_classify)

    _register(client, email="a@example.com", name="A")
    area = _create_area(client)
    client.post("/api/auth/logout")
    _register(client, email="b@example.com", name="B")

    response = client.post(
        "/api/classify",
        json={
            "polygon": POLYGON,
            "satellite": "sentinel2",
            "classifier": "auto",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "area_id": area["id"],
        },
    )

    assert response.status_code == 404
    assert db.query(Analysis).count() == 0
