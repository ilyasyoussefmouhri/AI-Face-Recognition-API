# env, settings
from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import ClassVar
from dataclasses import dataclass
from enum import Enum

class Settings(BaseSettings):
    # This automatically looks for DATABASE_URL in .env
    DATABASE_URL: PostgresDsn = Field(
        ...,
        validation_alias="DATABASE_URL",
        description="PostgreSQL SQLAlchemy URL"
    )

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def validate_driver(cls, v: PostgresDsn) -> str:
        url = str(v)
        # Ensure we use a modern driver if your setup requires it
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    PROJECT_ROOT: ClassVar[Path] = Path(__file__).resolve().parents[2]
    ENV_FILE: ClassVar[Path] = PROJECT_ROOT / ".env"
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

# Custom Exceptions
class NoFaceDetectedError(Exception):
    """Raised when no face is detected in the image."""
    pass


class MultipleFacesDetectedError(Exception):
    """Raised when multiple faces are detected in the image."""
    def __init__(self, num_faces: int):
        self.num_faces = num_faces
        super().__init__(f"Expected 1 face, but detected {num_faces} faces")

class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass

# Enum for device selection
class Device(str, Enum):
    CPU = "CPU"
    GPU = "GPU"


@dataclass
class ImageConfig:
    max_dimensions: tuple[int, int] = (4096, 4096)
    allowed_formats: set[str] = frozenset({'JPEG', 'PNG', 'WEBP', 'GIF'})
    max_image_pixels: int = 178956970
    verify_format: bool = True


# Usage
if __name__ == "__main__":
    # Only print when running this module directly (avoids noisy imports in servers/tests)
    settings = Settings()
    print(settings.DATABASE_URL)