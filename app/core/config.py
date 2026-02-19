# env, settings
from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union
from dataclasses import dataclass
from enum import Enum
from app.core.logs import logger
import json

class Settings(BaseSettings):
    # This automatically looks for DATABASE_URL in .env
    DATABASE_URL: PostgresDsn = Field(
        ...,
        validation_alias="DATABASE_URL",
        description="PostgreSQL SQLAlchemy URL"
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_driver(cls, v: PostgresDsn) -> str:
        url = str(v)
        # Ensure we use a modern driver if your setup requires it
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    SIGNATURES_JSON : Union[dict, str] = Field(
        ...,
        description="Magical numbers of allowed file types"
    )

    @field_validator("SIGNATURES_JSON", mode="after")
    @classmethod
    def parse_signatures(cls, v: str | dict) -> dict[str, tuple[bytes, ...]]:
        """Convert JSON hex strings to bytes tuples."""
        try:
            parsed = v if isinstance(v, dict) else json.loads(v)
            return {
                file_type: tuple(bytes.fromhex(sig) for sig in signatures)
                for file_type, signatures in parsed.items()
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse SIGNATURES: {e}")
            raise ValueError(f"Invalid SIGNATURES format: {e}")

    @property
    def SIGNATURES(self) -> dict[str, tuple[bytes, ...]]:
        """Access parsed signatures"""
        return self.SIGNATURES_JSON

    SECRET_KEY: str = Field(..., description="Secret key for cryptographic operations")
    ALGORITHM: str = Field(default="HS256", description="Allowed JWT algorithms")
    ACCESS_TOKEN_EXPIRE_SECONDS: int = Field(default=30*60, description="Access token expiration time in minutes")
    MAX_FILE_SIZE : int = Field(default=10485760, description="Maximum file size in bytes")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


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
settings = Settings()

if __name__ == "__main__":
    # Only print when running this module directly (avoids noisy imports in servers/tests)
    print(settings.DATABASE_URL)
