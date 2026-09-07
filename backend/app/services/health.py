"""Health check service."""

from backend.app.dependencies.models import get_gee


async def check_health() -> dict:
    """Check system health."""
    gee_ok = False
    try:
        get_gee()
        gee_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if gee_ok else "degraded",
        "gee_initialized": gee_ok,
    }