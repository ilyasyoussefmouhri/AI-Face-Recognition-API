# cosine similarity, thresholds
import numpy as np
from app.core.logs import logger


class InsightFaceMatcher:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        if e1.shape != e2.shape:
            logger.error(f"Embeddings must have same shape, got {e1.shape} and {e2.shape}")
            raise ValueError("Embeddings must have same shape")
        return float(np.dot(e1, e2))

    def match(self, e1: np.ndarray, e2: np.ndarray) -> bool:
        return self.similarity(e1, e2) >= self.threshold
