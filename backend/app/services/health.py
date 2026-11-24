"""Health check service."""

import psutil

from backend.app.dependencies.models import get_gee

_PROCESS = psutil.Process()


def memory_snapshot() -> dict[str, float]:
    """Retorna o consumo atual do processo e a memória disponível do sistema.

    Os valores são em MB e servem para acompanhar o pico de memória durante
    a classificação (RF + SVM + U-Net carregados ao mesmo tempo).
    """
    return {
        "rss_mb": round(_PROCESS.memory_info().rss / (1024 * 1024), 1),
        "system_available_mb": round(
            psutil.virtual_memory().available / (1024 * 1024), 1
        ),
    }


async def check_liveness() -> dict:
    """Verifica apenas se o processo está de pé (sem depender do GEE)."""
    return {"status": "ok", **memory_snapshot()}


async def check_health() -> dict:
    """Check completo: GEE + memória."""
    gee_ok = False
    try:
        get_gee()
        gee_ok = True
    except Exception:  # noqa: BLE001,S110
        pass

    return {
        "status": "healthy" if gee_ok else "degraded",
        "gee_initialized": gee_ok,
        **memory_snapshot(),
    }
