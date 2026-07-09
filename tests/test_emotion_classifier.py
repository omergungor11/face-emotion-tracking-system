"""src/emotion_classifier.py icin birim testleri."""

from src import config as cfg
from src.emotion_classifier import EmotionClassifier
from src.features import FaceFeatures


def make_classifier() -> EmotionClassifier:
    return EmotionClassifier(cfg)


def test_neutral_when_no_deltas():
    clf = make_classifier()
    baseline = FaceFeatures(ear=0.30, mar=0.10, smile=0.0, brow=0.20)
    features = FaceFeatures(ear=0.30, mar=0.10, smile=0.0, brow=0.20)

    label, probs = clf.classify(features, baseline)

    assert label == "Neutral"
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_happy_when_smile_delta_positive():
    clf = make_classifier()
    baseline = FaceFeatures(ear=0.30, mar=0.10, smile=0.0, brow=0.20)
    features = FaceFeatures(ear=0.30, mar=0.10, smile=0.5, brow=0.20)

    label, _ = clf.classify(features, baseline)

    assert label == "Happy"


def test_sad_when_smile_delta_negative():
    clf = make_classifier()
    baseline = FaceFeatures(ear=0.30, mar=0.10, smile=0.0, brow=0.20)
    features = FaceFeatures(ear=0.30, mar=0.10, smile=-0.5, brow=0.20)

    label, _ = clf.classify(features, baseline)

    assert label == "Sad"


def test_surprised_when_brow_mar_ear_up():
    clf = make_classifier()
    baseline = FaceFeatures(ear=0.30, mar=0.10, smile=0.0, brow=0.20)
    features = FaceFeatures(ear=0.50, mar=0.40, smile=0.0, brow=0.40)

    label, _ = clf.classify(features, baseline)

    assert label == "Surprised"


def test_smoothing_uses_recent_history():
    clf = make_classifier()
    baseline = FaceFeatures(ear=0.30, mar=0.10, smile=0.0, brow=0.20)
    happy = FaceFeatures(ear=0.30, mar=0.10, smile=0.5, brow=0.20)
    neutral = FaceFeatures(ear=0.30, mar=0.10, smile=0.0, brow=0.20)

    for _ in range(cfg.SMOOTHING_WINDOW):
        clf.classify(happy, baseline)

    label, probs_after_happy_streak = clf.classify(neutral, baseline)

    # tek bir notr kare, gecmisteki mutlu akisini aninda sifirlamamali
    assert probs_after_happy_streak["Happy"] > 0.1
