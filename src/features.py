"""Landmark koordinatlarindan olcek-bagimsiz geometrik ozellik cikarimi."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FaceFeatures:
    ear: float
    mar: float
    smile: float
    brow: float

    def as_array(self) -> np.ndarray:
        return np.array([self.ear, self.mar, self.smile, self.brow])

    @staticmethod
    def from_array(arr) -> "FaceFeatures":
        return FaceFeatures(*arr)

    def __sub__(self, other: "FaceFeatures") -> "FaceFeatures":
        return FaceFeatures.from_array(self.as_array() - other.as_array())


def _xy(landmarks, index: int) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x, lm.y])


def _dist(landmarks, i: int, j: int) -> float:
    return float(np.linalg.norm(_xy(landmarks, i) - _xy(landmarks, j)))


def interocular_distance(landmarks, left_idx: int, right_idx: int) -> float:
    return _dist(landmarks, left_idx, right_idx)


def eye_aspect_ratio(landmarks, eye_indices) -> float:
    p1, p2, p3, p4, p5, p6 = eye_indices
    vertical = _dist(landmarks, p2, p6) + _dist(landmarks, p3, p5)
    horizontal = _dist(landmarks, p1, p4)
    return vertical / (2.0 * horizontal + 1e-6)


def mouth_aspect_ratio(landmarks, top: int, bottom: int, left: int, right: int) -> float:
    vertical = _dist(landmarks, top, bottom)
    horizontal = _dist(landmarks, left, right)
    return vertical / (horizontal + 1e-6)


def smile_index(landmarks, left_corner: int, right_corner: int, inner_top: int, inner_bottom: int, norm: float) -> float:
    corner_y = (_xy(landmarks, left_corner)[1] + _xy(landmarks, right_corner)[1]) / 2.0
    center_y = (_xy(landmarks, inner_top)[1] + _xy(landmarks, inner_bottom)[1]) / 2.0
    # image koordinatlarinda y asagi dogru artar: kose merkeze gore yukarida
    # (kucuk y) ise pozitif deger dondurur -> gulumseme sinyali
    return (center_y - corner_y) / (norm + 1e-6)


def brow_raise(landmarks, brow_point: int, eye_top_point: int, norm: float) -> float:
    return _dist(landmarks, brow_point, eye_top_point) / (norm + 1e-6)


def extract_features(landmarks, cfg) -> FaceFeatures:
    norm = interocular_distance(landmarks, cfg.INTEROCULAR_LEFT, cfg.INTEROCULAR_RIGHT)

    ear = (
        eye_aspect_ratio(landmarks, cfg.RIGHT_EYE)
        + eye_aspect_ratio(landmarks, cfg.LEFT_EYE)
    ) / 2.0

    mar = mouth_aspect_ratio(
        landmarks, cfg.MOUTH_INNER_TOP, cfg.MOUTH_INNER_BOTTOM,
        cfg.MOUTH_LEFT_CORNER, cfg.MOUTH_RIGHT_CORNER,
    )

    smile = smile_index(
        landmarks, cfg.MOUTH_LEFT_CORNER, cfg.MOUTH_RIGHT_CORNER,
        cfg.MOUTH_INNER_TOP, cfg.MOUTH_INNER_BOTTOM, norm,
    )

    brow = (
        brow_raise(landmarks, cfg.RIGHT_EYEBROW_MID, cfg.RIGHT_EYE_TOP, norm)
        + brow_raise(landmarks, cfg.LEFT_EYEBROW_MID, cfg.LEFT_EYE_TOP, norm)
    ) / 2.0

    return FaceFeatures(ear=ear, mar=mar, smile=smile, brow=brow)
