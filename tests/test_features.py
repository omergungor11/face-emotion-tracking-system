"""src/features.py icin webcam/MediaPipe gerektirmeyen birim testleri."""

import math

from src import config as cfg
from src.features import (
    brow_raise,
    eye_aspect_ratio,
    extract_features,
    mouth_aspect_ratio,
    smile_index,
)


class FakeLandmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def build_landmarks(overrides: dict) -> list:
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(478)]
    for idx, (x, y) in overrides.items():
        landmarks[idx] = FakeLandmark(x, y)
    return landmarks


def test_eye_aspect_ratio_open_vs_closed():
    p1, p2, p3, p4, p5, p6 = cfg.RIGHT_EYE
    open_eye = build_landmarks({
        p1: (0.30, 0.40), p4: (0.40, 0.40),
        p2: (0.32, 0.36), p3: (0.38, 0.36),
        p6: (0.32, 0.44), p5: (0.38, 0.44),
    })
    closed_eye = build_landmarks({
        p1: (0.30, 0.40), p4: (0.40, 0.40),
        p2: (0.32, 0.395), p3: (0.38, 0.395),
        p6: (0.32, 0.405), p5: (0.38, 0.405),
    })
    assert eye_aspect_ratio(open_eye, cfg.RIGHT_EYE) > eye_aspect_ratio(closed_eye, cfg.RIGHT_EYE)


def test_mouth_aspect_ratio_open_vs_closed():
    open_mouth = build_landmarks({
        cfg.MOUTH_LEFT_CORNER: (0.40, 0.60), cfg.MOUTH_RIGHT_CORNER: (0.60, 0.60),
        cfg.MOUTH_INNER_TOP: (0.50, 0.55), cfg.MOUTH_INNER_BOTTOM: (0.50, 0.68),
    })
    closed_mouth = build_landmarks({
        cfg.MOUTH_LEFT_CORNER: (0.40, 0.60), cfg.MOUTH_RIGHT_CORNER: (0.60, 0.60),
        cfg.MOUTH_INNER_TOP: (0.50, 0.59), cfg.MOUTH_INNER_BOTTOM: (0.50, 0.61),
    })
    args = (cfg.MOUTH_INNER_TOP, cfg.MOUTH_INNER_BOTTOM, cfg.MOUTH_LEFT_CORNER, cfg.MOUTH_RIGHT_CORNER)
    assert mouth_aspect_ratio(open_mouth, *args) > mouth_aspect_ratio(closed_mouth, *args)


def test_smile_index_higher_when_corners_raised():
    neutral = build_landmarks({
        cfg.MOUTH_LEFT_CORNER: (0.40, 0.60), cfg.MOUTH_RIGHT_CORNER: (0.60, 0.60),
        cfg.MOUTH_INNER_TOP: (0.50, 0.58), cfg.MOUTH_INNER_BOTTOM: (0.50, 0.62),
    })
    smiling = build_landmarks({
        cfg.MOUTH_LEFT_CORNER: (0.40, 0.50), cfg.MOUTH_RIGHT_CORNER: (0.60, 0.50),
        cfg.MOUTH_INNER_TOP: (0.50, 0.58), cfg.MOUTH_INNER_BOTTOM: (0.50, 0.62),
    })
    args = (cfg.MOUTH_LEFT_CORNER, cfg.MOUTH_RIGHT_CORNER, cfg.MOUTH_INNER_TOP, cfg.MOUTH_INNER_BOTTOM)
    assert smile_index(smiling, *args, norm=1.0) > smile_index(neutral, *args, norm=1.0)


def test_smile_index_lower_when_corners_dropped():
    neutral = build_landmarks({
        cfg.MOUTH_LEFT_CORNER: (0.40, 0.60), cfg.MOUTH_RIGHT_CORNER: (0.60, 0.60),
        cfg.MOUTH_INNER_TOP: (0.50, 0.58), cfg.MOUTH_INNER_BOTTOM: (0.50, 0.62),
    })
    frowning = build_landmarks({
        cfg.MOUTH_LEFT_CORNER: (0.40, 0.68), cfg.MOUTH_RIGHT_CORNER: (0.60, 0.68),
        cfg.MOUTH_INNER_TOP: (0.50, 0.58), cfg.MOUTH_INNER_BOTTOM: (0.50, 0.62),
    })
    args = (cfg.MOUTH_LEFT_CORNER, cfg.MOUTH_RIGHT_CORNER, cfg.MOUTH_INNER_TOP, cfg.MOUTH_INNER_BOTTOM)
    assert smile_index(frowning, *args, norm=1.0) < smile_index(neutral, *args, norm=1.0)


def test_brow_raise_increases_with_distance():
    low_brow = build_landmarks({
        cfg.RIGHT_EYEBROW_MID: (0.35, 0.34), cfg.RIGHT_EYE_TOP: (0.35, 0.38),
    })
    raised_brow = build_landmarks({
        cfg.RIGHT_EYEBROW_MID: (0.35, 0.25), cfg.RIGHT_EYE_TOP: (0.35, 0.38),
    })
    args = (cfg.RIGHT_EYEBROW_MID, cfg.RIGHT_EYE_TOP)
    assert brow_raise(raised_brow, *args, norm=1.0) > brow_raise(low_brow, *args, norm=1.0)


def test_extract_features_returns_finite_values():
    overrides = {
        cfg.INTEROCULAR_LEFT: (0.30, 0.40),
        cfg.INTEROCULAR_RIGHT: (0.70, 0.40),
        cfg.MOUTH_LEFT_CORNER: (0.40, 0.60),
        cfg.MOUTH_RIGHT_CORNER: (0.60, 0.60),
        cfg.MOUTH_INNER_TOP: (0.50, 0.58),
        cfg.MOUTH_INNER_BOTTOM: (0.50, 0.62),
        cfg.RIGHT_EYEBROW_MID: (0.35, 0.30),
        cfg.RIGHT_EYE_TOP: (0.35, 0.35),
        cfg.LEFT_EYEBROW_MID: (0.65, 0.30),
        cfg.LEFT_EYE_TOP: (0.65, 0.35),
    }
    landmarks = build_landmarks(overrides)
    features = extract_features(landmarks, cfg)

    for value in features.as_array():
        assert math.isfinite(value)
