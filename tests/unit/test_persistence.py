"""Testes de áreas, histórico, persistência da classificação e outputs."""

import threading
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import get_settings
from backend.app.db.base import Base
from backend.app.db.models import Analysis
from backend.app.db.repositories import create_analysis
from backend.app.db.session import get_db
from backend.app.dependencies.models import get_gee
from backend.app.main import create_app
from backend.app.services.job_queue import (
    JobQueue,
    get_job_queue,
    mark_interrupted_jobs,
    run_analysis_job,
)

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

BOWTIE = {
    "type": "Polygon",
    "coordinates": [
        [
            [-50.0, -20.0],
            [-49.9, -19.9],
            [-49.9, -20.0],
            [-50.0, -19.9],
            [-50.0, -20.0],
        ]
    ],
}

BIG_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-50.0, -20.0],
            [-49.6, -20.0],
            [-49.6, -19.6],
            [-50.0, -19.6],
            [-50.0, -20.0],
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


class RecordingQueue:
    """Fila falsa: registra os jobs em vez de executá-los em outra thread."""

    def __init__(self) -> None:
        self.jobs: list[str] = []
        self.cancelled: list[str] = []
        self.cancel_result = True

    def submit(self, job_id: str) -> None:
        self.jobs.append(job_id)

    def cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return self.cancel_result and job_id in self.jobs


@pytest.fixture()
def queue():
    return RecordingQueue()


@pytest.fixture()
def client(db, queue):
    app = create_app()

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_gee] = lambda: None
    app.dependency_overrides[get_job_queue] = lambda: queue
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


def test_area_create_normalizes_open_ring(client):
    _register(client)
    open_polygon = {
        "type": "Polygon",
        "coordinates": [POLYGON["coordinates"][0][:-1]],
    }

    response = client.post(
        "/api/areas",
        json={"name": "Área aberta", "polygon": open_polygon},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    ring = body["geojson"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert body["area_ha"] > 0


def test_area_create_rejects_invalid_polygon(client):
    _register(client)

    response = client.post(
        "/api/areas",
        json={"name": "Auto-intersectada", "polygon": BOWTIE},
    )

    assert response.status_code == 422
    assert "auto-interseção" in response.json()["detail"]
    assert client.get("/api/areas").json() == []


def test_area_create_rejects_blank_name(client):
    _register(client)

    response = client.post(
        "/api/areas",
        json={"name": "   ", "polygon": POLYGON},
    )

    assert response.status_code == 422


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


# ------------------------------- relatório ---------------------------------


def _seed_report(db, user_id, tmp_path, job_id="job-report", **overrides):
    result = _fake_result(job_id)
    result["topography"] = {
        "dem_source": "NASADEM",
        "dem_resolution_m": 30,
        "elevation_min": 500.0,
        "elevation_max": 620.0,
        "elevation_mean": 560.0,
        "relief_m": 120.0,
        "slope_min": 0.1,
        "slope_mean": 4.2,
        "slope_max": 12.5,
        "dominant_class": "Suave ondulado",
        "classes": [
            {"class_name": "Plano", "area_ha": 3.0, "percent": 24.0},
            {"class_name": "Suave ondulado", "area_ha": 9.5, "percent": 76.0},
        ],
    }
    result["crop_stats"] = [
        {"class_name": "Soja", "area_ha": 8.0, "percent": 64.0},
        {"class_name": "Milho", "area_ha": 4.5, "percent": 36.0},
    ]
    analysis = _seed_analysis(
        db,
        user_id,
        job_id=job_id,
        result=result,
        **overrides,
    )

    from PIL import Image

    output_dir = Path(tmp_path) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "classification.png",
        "rgb.png",
        "overlay.png",
        "relief.png",
        "crop.png",
    ):
        Image.new("RGB", (80, 60), "green").save(output_dir / filename)
    return analysis


def test_report_pdf_for_owner(client, db, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_report(db, user["id"], tmp_path)

    response = client.get("/api/analyses/job-report/report.pdf")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/pdf")
    assert "georural_job-report.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 1000


def test_report_works_without_optional_images(client, db, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-sem-imagens")

    response = client.get("/api/analyses/job-sem-imagens/report.pdf")

    assert response.status_code == 200, response.text
    assert response.content.startswith(b"%PDF")


def test_report_requires_completed_analysis(client, db, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-running", status="running")
    _seed_analysis(db, user["id"], job_id="job-failed", status="failed")

    assert (
        client.get("/api/analyses/job-running/report.pdf").status_code == 409
    )
    assert (
        client.get("/api/analyses/job-failed/report.pdf").status_code == 409
    )


def test_report_is_protected_by_owner(client, db, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_report(db, user["id"], tmp_path, job_id="job-privado")

    client.post("/api/auth/logout")
    assert client.get("/api/analyses/job-privado/report.pdf").status_code == 401

    _register(client, email="outro@example.com", name="Outro")
    assert client.get("/api/analyses/job-privado/report.pdf").status_code == 404


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


CLASSIFY_PAYLOAD = {
    "polygon": POLYGON,
    "satellite": "sentinel2",
    "classifier": "auto",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
}


def _submit(client, **overrides):
    payload = {**CLASSIFY_PAYLOAD, **overrides}
    return client.post("/api/classify", json=payload)


def test_classify_rejects_inverted_dates(client, db, queue):
    _register(client)

    response = _submit(
        client, start_date="2023-12-31", end_date="2023-01-01"
    )

    assert response.status_code == 422
    assert queue.jobs == []
    assert db.query(Analysis).count() == 0


def test_classify_rejects_future_end_date(client, db, queue):
    _register(client)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    response = _submit(
        client, start_date="2023-01-01", end_date=tomorrow
    )

    assert response.status_code == 422
    assert queue.jobs == []


def test_classify_rejects_oversized_area(client, db, queue):
    _register(client)

    response = _submit(client, polygon=BIG_POLYGON)

    assert response.status_code == 422
    assert "excede o limite" in response.json()["detail"]
    assert queue.jobs == []
    assert db.query(Analysis).count() == 0


def test_classify_rejects_invalid_polygon(client, db, queue):
    _register(client)

    response = _submit(client, polygon=BOWTIE)

    assert response.status_code == 422
    assert "auto-interseção" in response.json()["detail"]
    assert queue.jobs == []
    assert db.query(Analysis).count() == 0


def test_classify_rejects_sentinel_only_classifier_for_landsat(
    client, db, queue
):
    _register(client)

    response = _submit(client, satellite="landsat8", classifier="unet_cnn")

    assert response.status_code == 422
    assert queue.jobs == []
    assert db.query(Analysis).count() == 0


def test_classify_normalizes_polygon_and_name(client, db, queue):
    _register(client)
    open_polygon = {
        "type": "Polygon",
        "coordinates": [POLYGON["coordinates"][0][:-1]],
    }

    response = _submit(
        client, polygon=open_polygon, name="  Safra validada  "
    )

    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    analysis = db.get(Analysis, job_id)
    assert analysis is not None
    assert analysis.name == "Safra validada"
    assert analysis.area_ha is not None
    assert analysis.area_ha > 0
    ring = analysis.geojson["coordinates"][0]
    assert ring[0] == ring[-1]


def test_classification_is_queued(client, db, queue):
    user = _register(client)
    area = _create_area(client)

    response = _submit(
        client, area_id=area["id"], name="Safra 2023"
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]
    assert queue.jobs == [job_id]

    analysis = db.get(Analysis, job_id)
    assert analysis is not None
    assert analysis.user_id == user["id"]
    assert analysis.area_id == area["id"]
    assert analysis.name == "Safra 2023"
    assert analysis.status == "queued"
    assert analysis.stage == "queued"
    assert analysis.progress == 0
    assert analysis.geojson == POLYGON


def test_completed_job_is_persisted(client, db, queue):
    _register(client)
    area = _create_area(client)
    response = _submit(client, area_id=area["id"], name="Safra 2023")
    job_id = response.json()["job_id"]

    def fake_classify(**kwargs):
        kwargs["on_progress"]("inference", 65)
        # O progresso é visível no banco enquanto a classificação roda.
        running = db.get(Analysis, job_id)
        assert running is not None
        assert running.status == "running"
        assert running.stage == "inference"
        assert running.progress == 65
        return _fake_result(kwargs["job_id"])

    run_analysis_job(job_id, db, classify=fake_classify)

    analysis = db.get(Analysis, job_id)
    assert analysis is not None
    assert analysis.status == "completed"
    assert analysis.stage == "done"
    assert analysis.progress == 100
    assert analysis.result["measurements"]["total_area_ha"] == 12.5
    assert analysis.area_ha == 12.5
    assert analysis.coverage == "brazil"

    detail = client.get(f"/api/analyses/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"
    assert detail.json()["area_name"] == "Fazenda"


def test_failed_job_is_persisted(client, db, queue):
    _register(client)
    response = _submit(client)
    job_id = response.json()["job_id"]

    run_analysis_job(
        job_id,
        db,
        classify=lambda **kwargs: _fake_result(
            kwargs["job_id"], status="failed"
        ),
    )

    analysis = db.get(Analysis, job_id)
    assert analysis is not None
    assert analysis.status == "failed"
    assert analysis.stage == "failed"
    assert analysis.error == "sem imagens"


def test_job_exception_marks_failed(client, db, queue):
    _register(client)
    response = _submit(client)
    job_id = response.json()["job_id"]

    def explode(**kwargs):
        raise RuntimeError("GEE fora do ar")

    run_analysis_job(job_id, db, classify=explode)

    analysis = db.get(Analysis, job_id)
    assert analysis is not None
    assert analysis.status == "failed"
    assert "GEE fora do ar" in (analysis.error or "")


def _release_spy(monkeypatch, enabled=True):
    from backend.app.services import job_queue

    released: list[bool] = []
    monkeypatch.setattr(
        job_queue, "release_model_cache", lambda: released.append(True)
    )
    monkeypatch.setattr(
        job_queue,
        "get_settings",
        lambda: SimpleNamespace(RELEASE_MODELS_AFTER_JOB=enabled),
    )
    return released


def test_job_releases_models_after_success(client, db, queue, monkeypatch):
    released = _release_spy(monkeypatch)
    _register(client)
    job_id = _submit(client).json()["job_id"]

    run_analysis_job(
        job_id, db, classify=lambda **kwargs: _fake_result(kwargs["job_id"])
    )

    assert released == [True]


def test_job_releases_models_after_failure(client, db, queue, monkeypatch):
    released = _release_spy(monkeypatch)
    _register(client)
    job_id = _submit(client).json()["job_id"]

    def explode(**kwargs):
        raise RuntimeError("sem memória")

    run_analysis_job(job_id, db, classify=explode)

    assert released == [True]


def test_job_keeps_models_when_release_is_disabled(
    client, db, queue, monkeypatch
):
    released = _release_spy(monkeypatch, enabled=False)
    _register(client)
    job_id = _submit(client).json()["job_id"]

    run_analysis_job(
        job_id, db, classify=lambda **kwargs: _fake_result(kwargs["job_id"])
    )

    assert released == []
    analysis = db.get(Analysis, job_id)
    assert analysis is not None
    assert analysis.status == "completed"


def test_classification_rejects_foreign_area(client, db, queue):
    _register(client, email="a@example.com", name="A")
    area = _create_area(client)
    client.post("/api/auth/logout")
    _register(client, email="b@example.com", name="B")

    response = _submit(client, area_id=area["id"])

    assert response.status_code == 404
    assert queue.jobs == []
    assert db.query(Analysis).count() == 0


# ----------------------------- retry e órfãos ------------------------------


def _completed_analysis(client, db, queue, name="Safra 2023"):
    area = _create_area(client)
    response = _submit(client, area_id=area["id"], name=name)
    job_id = response.json()["job_id"]
    run_analysis_job(
        job_id, db, classify=lambda **kwargs: _fake_result(kwargs["job_id"])
    )
    return job_id, area["id"]


def test_retry_copies_parameters_and_enqueues(client, db, queue):
    _register(client)
    job_id, area_id = _completed_analysis(client, db, queue)

    response = client.post(f"/api/analyses/{job_id}/retry")

    assert response.status_code == 202, response.text
    new_id = response.json()["job_id"]
    assert new_id != job_id
    assert queue.jobs[-1] == new_id

    copy = db.get(Analysis, new_id)
    assert copy is not None
    assert copy.status == "queued"
    assert copy.stage == "queued"
    assert copy.progress == 0
    assert copy.area_id == area_id
    assert copy.geojson == POLYGON
    assert copy.name == "Safra 2023 (reprocessada)"
    assert copy.satellite == "sentinel2"
    assert copy.classifier == "auto"
    assert copy.start_date == "2023-01-01"


def test_retry_rejects_running_analysis(client, db, queue):
    _register(client)
    response = _submit(client)
    job_id = response.json()["job_id"]

    retry = client.post(f"/api/analyses/{job_id}/retry")

    assert retry.status_code == 409
    assert len(queue.jobs) == 1


def test_retry_rejects_foreign_analysis(client, db, queue):
    _register(client, email="a@example.com", name="A")
    job_id, _ = _completed_analysis(client, db, queue)
    client.post("/api/auth/logout")
    _register(client, email="b@example.com", name="B")

    response = client.post(f"/api/analyses/{job_id}/retry")

    assert response.status_code == 404


# ----------------------- exclusão de análises ------------------------------


def test_delete_completed_analysis_removes_outputs(
    client, db, tmp_path, monkeypatch
):
    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-delete")
    output_dir = Path(tmp_path) / "job-delete"
    output_dir.mkdir(parents=True)
    (output_dir / "classification.png").write_bytes(b"png")

    response = client.delete("/api/analyses/job-delete")

    assert response.status_code == 204
    assert db.get(Analysis, "job-delete") is None
    assert not output_dir.exists()
    assert client.get("/api/analyses/job-delete").status_code == 404


def test_delete_analysis_without_outputs(client, db, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-sem-arquivos")

    response = client.delete("/api/analyses/job-sem-arquivos")

    assert response.status_code == 204
    assert db.get(Analysis, "job-sem-arquivos") is None


def test_delete_failed_analysis(client, db):
    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-falha", status="failed")

    response = client.delete("/api/analyses/job-falha")

    assert response.status_code == 204
    assert db.get(Analysis, "job-falha") is None


def test_delete_queued_analysis_cancels_job(client, db, queue):
    _register(client)
    job_id = _submit(client).json()["job_id"]
    assert queue.jobs == [job_id]

    response = client.delete(f"/api/analyses/{job_id}")

    assert response.status_code == 204
    assert queue.cancelled == [job_id]
    assert db.get(Analysis, job_id) is None


def test_delete_queued_analysis_conflicts_when_started(client, db, queue):
    _register(client)
    job_id = _submit(client).json()["job_id"]
    queue.cancel_result = False

    response = client.delete(f"/api/analyses/{job_id}")

    assert response.status_code == 409
    assert db.get(Analysis, job_id) is not None


def test_delete_running_analysis_conflicts(client, db, queue):
    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-running-del", status="running")

    response = client.delete("/api/analyses/job-running-del")

    assert response.status_code == 409
    assert db.get(Analysis, "job-running-del") is not None


def test_delete_analysis_is_owner_scoped(client, db):
    first = _register(client, email="a@example.com", name="A")
    _seed_analysis(db, first["id"], job_id="job-dono")
    client.post("/api/auth/logout")
    _register(client, email="b@example.com", name="B")

    response = client.delete("/api/analyses/job-dono")

    assert response.status_code == 404
    assert db.get(Analysis, "job-dono") is not None


def test_delete_analysis_file_error_preserves_row(
    client, db, tmp_path, monkeypatch
):
    from backend.app.services import output_files

    settings = get_settings()
    monkeypatch.setattr(settings, "OUTPUTS_DIR", str(tmp_path))

    user = _register(client)
    _seed_analysis(db, user["id"], job_id="job-erro-arquivo")
    output_dir = Path(tmp_path) / "job-erro-arquivo"
    output_dir.mkdir(parents=True)

    def fail_rmtree(path):
        raise OSError("sem permissão")

    monkeypatch.setattr(output_files.shutil, "rmtree", fail_rmtree)

    response = client.delete("/api/analyses/job-erro-arquivo")

    assert response.status_code == 500
    assert db.get(Analysis, "job-erro-arquivo") is not None


def test_delete_missing_analysis(client, db):
    _register(client)

    assert client.delete("/api/analyses/inexistente").status_code == 404


def test_mark_interrupted_jobs(db):
    create_analysis(
        db,
        job_id="job-queued",
        satellite="sentinel2",
        classifier="auto",
        status="queued",
        stage="queued",
    )
    create_analysis(
        db,
        job_id="job-running",
        satellite="sentinel2",
        classifier="auto",
        status="running",
        stage="inference",
    )
    create_analysis(
        db,
        job_id="job-done",
        satellite="sentinel2",
        classifier="auto",
        status="completed",
        stage="done",
    )

    assert mark_interrupted_jobs(db) == 2

    for job_id in ("job-queued", "job-running"):
        analysis = db.get(Analysis, job_id)
        assert analysis is not None
        assert analysis.status == "failed"
        assert analysis.stage == "interrupted"

    done = db.get(Analysis, "job-done")
    assert done is not None
    assert done.status == "completed"


def test_job_queue_runs_submitted_job():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with session_factory() as session:
        create_analysis(
            session,
            job_id="job-queue",
            satellite="sentinel2",
            classifier="auto",
            geojson=POLYGON,
            start_date="2023-01-01",
            end_date="2023-12-31",
            status="queued",
            stage="queued",
        )

    started = threading.Event()

    def fake_classify(**kwargs):
        started.set()
        return _fake_result(kwargs["job_id"])

    queue = JobQueue(
        max_workers=1,
        session_factory=session_factory,
        classify=fake_classify,
    )
    queue.submit("job-queue")

    assert started.wait(timeout=10)
    queue.shutdown(wait=True)

    with session_factory() as session:
        analysis = session.get(Analysis, "job-queue")
        assert analysis is not None
        assert analysis.status == "completed"
        assert analysis.progress == 100

    Base.metadata.drop_all(engine)


def test_job_queue_cancel_unknown_job():
    queue = JobQueue(max_workers=1)

    assert queue.cancel("nao-existe") is False
    queue.shutdown()


def test_job_queue_cancels_pending_job():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with session_factory() as session:
        for job_id in ("job-primeiro", "job-segundo"):
            create_analysis(
                session,
                job_id=job_id,
                satellite="sentinel2",
                classifier="auto",
                geojson=POLYGON,
                start_date="2023-01-01",
                end_date="2023-12-31",
                status="queued",
                stage="queued",
            )

    started = threading.Event()
    release = threading.Event()

    def fake_classify(**kwargs):
        if kwargs["job_id"] == "job-primeiro":
            started.set()
            assert release.wait(timeout=10)
        return _fake_result(kwargs["job_id"])

    queue = JobQueue(
        max_workers=1,
        session_factory=session_factory,
        classify=fake_classify,
    )
    queue.submit("job-primeiro")
    assert started.wait(timeout=10)

    queue.submit("job-segundo")
    assert queue.cancel("job-segundo") is True
    assert queue.cancel("job-primeiro") is False

    release.set()
    queue.shutdown(wait=True)

    with session_factory() as session:
        first = session.get(Analysis, "job-primeiro")
        second = session.get(Analysis, "job-segundo")
        assert first is not None and first.status == "completed"
        assert second is not None and second.status == "queued"

    Base.metadata.drop_all(engine)
