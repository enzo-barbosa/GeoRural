"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import get_settings
from backend.app.routers import (
    analyses,
    areas,
    auth,
    classify,
    geocode,
    health,
    outputs,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    Path(settings.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown (cleanup if needed)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="GeoRural API",
        description="API para classificação de uso e cobertura do solo em propriedades rurais",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(areas.router)
    app.include_router(analyses.router)
    app.include_router(classify.router)
    app.include_router(geocode.router)
    app.include_router(outputs.router)

    # Serve React frontend (build directory)
    frontend_dist = Path("backend/frontend/dist")
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_react(full_path: str):
            # Skip API and protected output routes
            if full_path.startswith(("api/", "outputs/")):
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Not found")
            index_file = frontend_dist / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return {"detail": "Frontend not built. Run 'npm run build' in frontend directory."}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=True,
    )