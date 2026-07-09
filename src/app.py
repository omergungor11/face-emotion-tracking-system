"""Ana uygulama dongusu: kamera -> face mesh -> features -> classifier -> HUD."""

import time
from pathlib import Path

import cv2

from . import config as cfg
from .calibration import Calibrator, load_baseline, save_baseline
from .emotion_classifier import EmotionClassifier
from .face_mesh import FaceMeshDetector
from .features import extract_features
from .hud import HUD


class App:
    def __init__(self, camera_index: int = 0, calibration_path: Path | None = None):
        self.camera_index = camera_index
        self.calibration_path = calibration_path or Path("calibration_profiles/default.json")
        self.detector = FaceMeshDetector()
        self.calibrator = Calibrator(cfg.CALIBRATION_DURATION_SEC, cfg.CALIBRATION_MIN_FRAMES)
        self.classifier = EmotionClassifier(cfg)
        self.hud = HUD(cfg)
        self.baseline = load_baseline(self.calibration_path)
        self.show_mesh = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Kamera acilamadi (index={self.camera_index}). "
                "Baska bir uygulama tarafindan kullaniliyor olabilir."
            )

        if self.baseline is None:
            self.calibrator.start()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = self.detector.process(rgb)

                if self.calibrator.is_calibrating:
                    frame = self._handle_calibration(frame, landmarks)
                else:
                    frame = self._handle_tracking(frame, landmarks)

                cv2.imshow("Face Emotion Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("c"):
                    self.calibrator.start()
                elif key == ord("m"):
                    self.show_mesh = not self.show_mesh
                elif key == ord("s"):
                    self._save_screenshot(frame)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

    def _handle_calibration(self, frame, landmarks):
        if landmarks is not None:
            features = extract_features(landmarks, cfg)
            self.calibrator.add_sample(features)
        frame = self.hud.draw_calibration_overlay(frame, self.calibrator.progress(), landmarks is not None)
        if self.calibrator.is_done():
            self.baseline = self.calibrator.finalize()
            save_baseline(self.baseline, self.calibration_path)
            self.classifier.reset()
        return frame

    def _handle_tracking(self, frame, landmarks):
        fps = self.hud.tick_fps()
        if landmarks is None:
            return self.hud.draw_no_face_warning(frame)

        features = extract_features(landmarks, cfg)
        label, probs = self.classifier.classify(features, self.baseline)
        self.hud.record_emotion(label)

        if self.show_mesh:
            frame = self.hud.draw_landmarks(frame, landmarks)
        return self.hud.draw_main_overlay(frame, label, probs, fps)

    def _save_screenshot(self, frame) -> None:
        out_dir = Path("assets/screenshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"screenshot_{int(time.time())}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Screenshot kaydedildi: {filename}")
