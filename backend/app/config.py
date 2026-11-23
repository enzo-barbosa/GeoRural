"""Backend configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Google Earth Engine
    GEE_PROJECT_ID: str

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Model paths
    MODELS_DIR: str = "models"
    RF_MODEL_PATH: str = "models/random_forest.pkl"
    SVM_MODEL_PATH: str = "models/svm.pkl"
    UNET_MODEL_PATH: str = "models/unet_model.pth"

    # Output directory
    OUTPUTS_DIR: str = "outputs"

    # Classification
    DEFAULT_PATCH_SIZE: int = 64
    DEFAULT_STRIDE: int = 32
    MAX_PIXELS_DOWNLOAD: int = 1_000_000_000
    # Limites para não estourar a memória nem o getDownloadURL do GEE.
    MAX_CLASSIFY_PIXELS: int = 6_000_000
    MAX_DOWNLOAD_BYTES: int = 40_000_000


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()