"""OpenCV uzerinde canli HUD (etiket, olasilik bar'lari, timeline, FPS) cizimi."""

import time
from collections import deque

import cv2
import numpy as np


class HUD:
    def __init__(self, cfg, history_seconds: int = 10, fps_estimate: int = 30):
        self.cfg = cfg
        self._emotion_history: deque = deque(maxlen=history_seconds * fps_estimate)
        self._fps_history: deque = deque(maxlen=30)
        self._last_time = time.time()

    def tick_fps(self) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_history.append(fps)
        return float(np.mean(self._fps_history))

    def record_emotion(self, label: str) -> None:
        self._emotion_history.append(label)

    def draw_calibration_overlay(self, frame, progress: float, face_found: bool):
        h, w = frame.shape[:2]
        text = "Lutfen notr bir ifade takinin..." if face_found else "Yuzunuzu kameraya gosterin"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        bar_w = int(w * 0.6)
        x0, y0 = 30, 70
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 20), (80, 80, 80), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * progress), y0 + 20), (0, 200, 0), -1)
        return frame

    def draw_main_overlay(self, frame, label: str, probs: dict, fps: float):
        h, w = frame.shape[:2]
        color = self.cfg.EMOTION_COLORS.get(label, (255, 255, 255))
        cv2.putText(frame, label, (w - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        y = 70
        for emotion, p in probs.items():
            bar_color = self.cfg.EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.putText(frame, emotion, (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cv2.rectangle(frame, (100, y), (220, y + 14), (60, 60, 60), -1)
            cv2.rectangle(frame, (100, y), (100 + int(120 * p), y + 14), bar_color, -1)
            y += 22

        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self._draw_timeline(frame)
        return frame

    def _draw_timeline(self, frame) -> None:
        h, w = frame.shape[:2]
        n = len(self._emotion_history)
        if n == 0:
            return
        base_y = h - 40
        chart_w = w - 40
        step = max(1, chart_w // max(n, 1))
        for i, label in enumerate(self._emotion_history):
            color = self.cfg.EMOTION_COLORS.get(label, (255, 255, 255))
            x = 20 + i * step
            cv2.line(frame, (x, base_y), (x, base_y - 15), color, 2)

    def draw_landmarks(self, frame, landmarks):
        h, w = frame.shape[:2]
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame

    def draw_no_face_warning(self, frame):
        cv2.putText(frame, "Yuz algilanamadi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
