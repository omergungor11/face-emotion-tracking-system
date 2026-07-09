"""Kalibre edilmis baseline'a gore kural-tabanli duygu siniflandirmasi."""

from collections import deque

import numpy as np

from .features import FaceFeatures

EMOTIONS = ["Neutral", "Happy", "Sad", "Surprised"]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class EmotionClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self._history: deque = deque(maxlen=cfg.SMOOTHING_WINDOW)

    def reset(self) -> None:
        self._history.clear()

    def classify(self, features: FaceFeatures, baseline: FaceFeatures):
        delta = features - baseline
        cfg = self.cfg

        happy_score = max(0.0, delta.smile) * cfg.W_SMILE
        sad_score = max(0.0, -delta.smile) * cfg.W_SMILE
        surprised_score = (
            max(0.0, delta.brow) * cfg.W_BROW
            + max(0.0, delta.mar) * cfg.W_MAR
            + max(0.0, delta.ear) * cfg.W_EAR
        ) / 3.0
        neutral_score = cfg.NEUTRAL_BIAS

        raw_scores = np.array([neutral_score, happy_score, sad_score, surprised_score])
        probs = _softmax(raw_scores / cfg.TEMPERATURE)

        self._history.append(probs)
        smoothed = np.mean(self._history, axis=0)

        label = EMOTIONS[int(np.argmax(smoothed))]
        return label, dict(zip(EMOTIONS, smoothed))
