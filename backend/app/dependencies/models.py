"""FastAPI dependencies for GEE and model management."""

import json
import os
from functools import lru_cache
from pathlib import Path

import ee
import joblib
import torch
from fastapi import Depends, HTTPException

from backend.app.config import get_settings
from src.classifiers.random_forest import load_model as load_rf
from src.classifiers.svm import load_model as load_svm
from src.classifiers.unet_cnn import build_unet, load_model as load_unet
from src.gee.auth import initialize_gee

# Global flag for GEE initialization
_gee_initialized = False


def get_gee() -> None:
    """Initialize GEE on first request."""
    global _gee_initialized
    if not _gee_initialized:
        try:
            initialize_gee()
            _gee_initialized = True
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"GEE initialization failed: {exc}") from exc


def get_model_path(model_name: str) -> Path:
    """Get model file path."""
    settings = get_settings()
    return Path(settings.MODELS_DIR) / {
        "random_forest": "random_forest.pkl",
        "svm": "svm.pkl",
        "unet_cnn": "unet_model.pth",
    }[model_name]


def check_model_exists(model_name: str) -> bool:
    """Check if model file exists."""
    return get_model_path(model_name).exists()


@lru_cache(maxsize=3)
def get_cached_model(model_name: str):
    """Load and cache model in memory."""
    path = get_model_path(model_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found at {path}")

    if model_name == "random_forest":
        return load_rf(str(path))
    if model_name == "svm":
        return load_svm(str(path))
    if model_name == "unet_cnn":
        model = build_unet(input_shape=(64, 64, 9), num_classes=5)
        return load_unet(model, str(path))

    raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")


def get_models_status() -> dict[str, bool]:
    """Get availability status of all models."""
    return {
        "random_forest": check_model_exists("random_forest"),
        "svm": check_model_exists("svm"),
        "unet_cnn": check_model_exists("unet_cnn"),
    }