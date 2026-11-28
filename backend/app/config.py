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

    # Banco de dados (Postgres no Docker Compose; localhost:5434 no host)
    DATABASE_URL: str = (
        "postgresql+psycopg://georural:georural@localhost:5434/georural"
    )

    # Autenticação
    JWT_SECRET_KEY: str = "dev-insecure-secret-change-me-please-32-bytes"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    AUTH_COOKIE_NAME: str = "georural_access"
    AUTH_COOKIE_SECURE: bool = False

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

    # Inferência: processar em chunks reduz o pico de memória do ensemble.
    INFERENCE_CHUNK_SIZE: int = 200_000
    UNET_BATCH_SIZE: int = 8

    # Fila de classificação (1 worker em processo). Na inicialização, análises
    # que ficaram "queued"/"running" após um restart são marcadas como falhas.
    RECOVER_STALE_JOBS: bool = True

    # Libera os modelos carregados ao final de cada classificação. Evita manter
    # vários GB de RAM ociosos; a próxima análise recarrega do disco.
    RELEASE_MODELS_AFTER_JOB: bool = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()