import logging
import logging.config
from .config import settings


def setup_logging():
    """Setup logging configuration"""

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.log_format,
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": "logs/app.log",
                "mode": "a",
            },
        },
        "loggers": {
            "": {
                "level": settings.log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "src": {
                "level": settings.log_level,
                "handlers": ["console", "file"] if settings.environment == "production" else ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "celery": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }

    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)

    logging.config.dictConfig(logging_config)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured for environment: {settings.environment}")
