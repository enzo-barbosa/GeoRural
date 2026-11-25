"""Fila em processo para executar uma classificação por vez."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Analysis
from backend.app.db.session import SessionLocal
from backend.app.services.classification_service import classify_area

logger = logging.getLogger(__name__)

Classifier = Callable[..., dict[str, Any]]

INTERRUPTED_ERROR = (
    "Servidor reiniciou durante a análise. Envie a classificação novamente."
)


def _mark_failed(db: Session, job_id: str, error: str) -> None:
    """Marca o job como falho, mesmo após um rollback."""
    try:
        db.rollback()
    except Exception:
        logger.exception("Falha ao limpar transação do job %s", job_id)

    analysis = db.get(Analysis, job_id)
    if analysis is None:
        return
    analysis.status = "failed"
    analysis.stage = "failed"
    analysis.error = error
    db.commit()


def _mark_completed(db: Session, analysis: Analysis, result: dict) -> None:
    """Persiste o resultado da classificação concluída."""
    measurements = result.get("measurements") or {}
    analysis.status = "completed"
    analysis.stage = "done"
    analysis.progress = 100
    analysis.coverage = result.get("coverage")
    analysis.label_source = result.get("label_source")
    analysis.num_images = result.get("num_images")
    analysis.confidence_mean = result.get("confidence_mean")
    analysis.models_used = result.get("models_used")
    analysis.weights = result.get("weights")
    analysis.area_ha = measurements.get("total_area_ha")
    analysis.result = result
    db.commit()


def _update_progress(
    db: Session, analysis: Analysis, stage: str, progress: int
) -> None:
    """Atualiza etapa/progresso; erro de banco não interrompe a classificação."""
    try:
        analysis.status = "running"
        analysis.stage = stage
        analysis.progress = max(0, min(100, int(progress)))
        db.commit()
    except Exception:
        logger.exception("Falha ao atualizar progresso do job %s", analysis.id)
        db.rollback()


def run_analysis_job(
    job_id: str,
    db: Session,
    *,
    classify: Classifier = classify_area,
) -> None:
    """Executa o job já enfileirado e persiste o estado final.

    Recebe a sessão para ser testável sem depender da thread da fila.
    """
    analysis = db.get(Analysis, job_id)
    if analysis is None:
        logger.warning("Job %s não encontrado no banco", job_id)
        return
    if analysis.status != "queued":
        logger.info(
            "Job %s ignorado (status atual: %s)", job_id, analysis.status
        )
        return

    analysis.status = "running"
    analysis.stage = "starting"
    analysis.progress = 0
    db.commit()

    geojson = analysis.geojson or {}
    rings = geojson.get("coordinates") or [[]]
    polygon_coords = rings[0] if rings else []

    try:
        result = classify(
            polygon_coords=polygon_coords,
            satellite=analysis.satellite,
            classifier_name=analysis.classifier,
            start_date=analysis.start_date or "",
            end_date=analysis.end_date or "",
            max_cloud=(
                analysis.max_cloud if analysis.max_cloud is not None else 80
            ),
            include_topography=analysis.include_topography,
            include_crop=analysis.include_crop,
            postprocess=analysis.postprocess,
            job_id=analysis.id,
            on_progress=lambda stage, progress: _update_progress(
                db, analysis, stage, progress
            ),
        )
    except Exception as exc:
        logger.exception("Job %s falhou durante a classificação", job_id)
        _mark_failed(db, job_id, f"Falha inesperada: {exc}")
        return

    if result.get("status") == "failed":
        _mark_failed(
            db, job_id, result.get("error", "Falha na classificação")
        )
        return

    try:
        _mark_completed(db, analysis, result)
    except Exception:
        logger.exception("Job %s falhou ao salvar o resultado", job_id)
        _mark_failed(db, job_id, "Falha ao salvar o resultado da análise")


def mark_interrupted_jobs(db: Session | None = None) -> int:
    """Marca jobs órfãos de um restart anterior como interrompidos."""
    owns_session = db is None
    session = db or SessionLocal()
    try:
        analyses = list(
            session.scalars(
                select(Analysis).where(
                    Analysis.status.in_(("queued", "running"))
                )
            ).all()
        )
        for analysis in analyses:
            analysis.status = "failed"
            analysis.stage = "interrupted"
            analysis.error = INTERRUPTED_ERROR
        if analyses:
            session.commit()
        return len(analyses)
    finally:
        if owns_session:
            session.close()


class JobQueue:
    """Fila com um único worker para não rodar duas inferências juntas."""

    def __init__(
        self,
        max_workers: int = 1,
        session_factory: Callable[[], Session] = SessionLocal,
        classify: Classifier = classify_area,
    ) -> None:
        self._session_factory = session_factory
        self._classify = classify
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="georural-job"
        )

    def submit(self, job_id: str) -> None:
        """Enfileira o job; o retorno é imediato."""
        self._executor.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        with self._session_factory() as db:
            try:
                run_analysis_job(job_id, db, classify=self._classify)
            except Exception:
                logger.exception("Falha inesperada no worker do job %s", job_id)
                _mark_failed(
                    db, job_id, "Falha inesperada no worker de classificação"
                )

    def shutdown(self, wait: bool = False) -> None:
        """Encerra a fila; jobs ainda na fila são descartados."""
        self._executor.shutdown(wait=wait, cancel_futures=True)


def get_job_queue(request: Request) -> JobQueue:
    """Dependência do FastAPI que entrega a fila da aplicação."""
    return request.app.state.job_queue


def enqueue_job(db: Session, analysis: Analysis, queue: JobQueue) -> None:
    """Enfileira a análise; se a fila falhar, marca o job como falho."""
    try:
        queue.submit(analysis.id)
    except Exception as exc:
        logger.exception("Falha ao enfileirar o job %s", analysis.id)
        analysis.status = "failed"
        analysis.stage = "failed"
        analysis.error = "Não foi possível enfileirar a análise."
        db.commit()
        raise RuntimeError(
            "Não foi possível enfileirar a análise."
        ) from exc
