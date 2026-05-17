from functools import lru_cache

import environ
import structlog
from dotenv import load_dotenv

logger = structlog.stdlib.get_logger(__name__)


@environ.config(frozen=True)
class Config:
    """Application configuration loaded from env vars and files with secrets."""

    @environ.config(frozen=True)
    class HuggingFaceConfig:
        token: str = environ.var(name="HF_TOKEN", help="HuggingFace API token")

    hf: HuggingFaceConfig = environ.group(HuggingFaceConfig)


@lru_cache
def load_config() -> Config:
    """Load env variables from .env file and cache parsed application configuration."""
    logger.info("Loading app config")
    _ = load_dotenv()
    config = environ.to_config(Config)
    logger.info("App config parsed and verified successfully")
    return config
