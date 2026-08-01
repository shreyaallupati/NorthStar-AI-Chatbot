from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
DB_PATH = DATA_DIR / "northstar.db"
INTENT_MODEL_PATH = DATA_DIR / "intent_model.joblib"
RETRIEVER_INDEX_PATH = DATA_DIR / "retriever_index.joblib"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "North Star Support Bot"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Minimum top-class probability required before the local classifier's
    # prediction is trusted. Anything below this routes to `fallback`.
    intent_confidence_threshold: float = 0.30

    # Minimum fused retrieval score for a catalog document to be considered a hit.
    retrieval_score_floor: float = 0.05

    @property
    def allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
