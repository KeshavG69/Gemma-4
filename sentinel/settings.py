from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    GEMMA_BASE_URL: str = ""
    GEMMA_MODEL: str = ""
    DB_PATH: str = ""
    CAPTURE_INTERVAL_SECONDS: int = 0
    MAX_IMAGE_WIDTH: int = 1280

    @property
    def db_path(self) -> Path:
        """DB_PATH resolved to an absolute path (relative to project root)."""
        
        return self.DB_PATH





def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()


settings = get_settings()