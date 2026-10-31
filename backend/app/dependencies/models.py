"""FastAPI dependencies for GEE and model management."""

from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from backend.app.config import get_settings
from src.classifiers.crop_classifier import load_model as load_crop
from src.classifiers.random_forest import load_model as load_rf
from src.classifiers.svm import load_model as load_svm
from src.classifiers.unet_cnn import build_unet
from src.classifiers.unet_cnn import load_model as load_unet
from src.config import NUM_CLASSES
from src.gee.auth import initialize_gee

# Global flag for GEE initialization
_gee_initialized = False

MODEL_FILES = {
    "brazil_rf": "brazil_rf.pkl",
    "brazil_svm": "brazil_svm.pkl",
    "brazil_unet": "brazil_unet.pth",
    "global_rf": "global_rf.pkl",
    "global_svm": "global_svm.pkl",
    "crop": "crop_classifier.pkl",
}

# Nomes antigos mantidos para compatibilidade com artefatos já treinados.
LEGACY_FILES = {
    "brazil_rf": "random_forest.pkl",
    "brazil_svm": "svm.pkl",
    "brazil_unet": "unet_model.pth",
}

ALIASES = {
    "random_forest": "brazil_rf",
    "svm": "brazil_svm",
    "unet_cnn": "brazil_unet",
}


def get_gee() -> None:
    """Initialize GEE on first request."""
    global _gee_initialized
    if not _gee_initialized:
        try:
            initialize_gee()
            _gee_initialized = True
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"GEE initialization failed: {exc}"
            ) from exc


def resolve_model_name(model_name: str) -> str:
    """Normaliza aliases antigos para os nomes por domínio."""
    return ALIASES.get(model_name, model_name)


def get_model_path(model_name: str) -> Path:
    """Get model file path (preferindo o nome novo, com fallback legado)."""
    name = resolve_model_name(model_name)
    if name not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    settings = get_settings()
    base = Path(settings.MODELS_DIR)
    primary = base / MODEL_FILES[name]
    if primary.exists() or name not in LEGACY_FILES:
        return primary

    legacy = base / LEGACY_FILES[name]
    return legacy if legacy.exists() else primary


def check_model_exists(model_name: str) -> bool:
    """Check if model file exists."""
    return get_model_path(model_name).exists()


@lru_cache(maxsize=8)
def get_cached_model(model_name: str):
    """Load and cache model in memory."""
    name = resolve_model_name(model_name)
    path = get_model_path(name)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Model {name} not found at {path}"
        )

    if name in {"brazil_rf", "global_rf"}:
        return load_rf(str(path))
    if name in {"brazil_svm", "global_svm"}:
        return load_svm(str(path))
    if name == "brazil_unet":
        model = build_unet(input_shape=(64, 64, 9), num_classes=NUM_CLASSES)
        return load_unet(model, str(path))
    if name == "crop":
        return load_crop(str(path))

    raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")


def get_models_status() -> dict[str, bool]:
    """Get availability status of all models."""
    return {name: check_model_exists(name) for name in MODEL_FILES}
