"""Lazy singleton loader for the trained model, scaler and metrics."""

import json
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

from src.predict import load_model_and_scaler

load_dotenv()

logger = logging.getLogger("finguard.api.model_loader")

MODEL_VERSION = "1.0"


class ModelBundle:
    """Holds everything the API needs to serve predictions."""

    def __init__(self):
        self.model, self.scaler = load_model_and_scaler()
        self.metrics = self._load_metrics()
        logger.info("Model bundle loaded (version %s)", MODEL_VERSION)

    @staticmethod
    def _load_metrics() -> dict:
        metrics_path = os.getenv("METRICS_PATH", "artifacts/metrics.json")
        if not os.path.exists(metrics_path):
            logger.warning("%s not found — /stats will return 503", metrics_path)
            return {}
        with open(metrics_path) as f:
            return json.load(f)


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    """Load artifacts once per process; FastAPI dependency-friendly."""
    return ModelBundle()
