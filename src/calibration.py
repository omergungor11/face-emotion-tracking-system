"""Kullaniciya ozel notr yuz baseline'inin yakalanmasi ve saklanmasi."""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .features import FaceFeatures


class Calibrator:
    def __init__(self, duration_sec: float, min_frames: int):
        self.duration_sec = duration_sec
        self.min_frames = min_frames
        self._samples: list[np.ndarray] = []
        self._start_time: Optional[float] = None

    def start(self) -> None:
        self._samples = []
        self._start_time = time.time()

    @property
    def is_calibrating(self) -> bool:
        return self._start_time is not None

    def progress(self) -> float:
        if self._start_time is None:
            return 1.0
        elapsed = time.time() - self._start_time
        return min(elapsed / self.duration_sec, 1.0)

    def add_sample(self, features: FaceFeatures) -> None:
        if self._start_time is None:
            return
        self._samples.append(features.as_array())

    def is_done(self) -> bool:
        return (
            self._start_time is not None
            and self.progress() >= 1.0
            and len(self._samples) >= self.min_frames
        )

    def finalize(self) -> FaceFeatures:
        arr = np.array(self._samples)
        median = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - median), axis=0) + 1e-6
        mask = np.all(np.abs(arr - median) < 3 * mad, axis=1)
        filtered = arr[mask] if mask.any() else arr
        baseline = FaceFeatures.from_array(filtered.mean(axis=0))
        self._start_time = None
        self._samples = []
        return baseline


def save_baseline(baseline: FaceFeatures, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"ear": baseline.ear, "mar": baseline.mar, "smile": baseline.smile, "brow": baseline.brow},
            f, indent=2,
        )


def load_baseline(path: Path) -> Optional[FaceFeatures]:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return FaceFeatures(**data)
