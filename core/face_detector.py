"""Face landmark detection using MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class FaceDetection:
    landmarks_norm: list[list[float]]
    landmarks_px: list[tuple[int, int]]
    bbox_xyxy: tuple[int, int, int, int]


class FaceDetector:
    """Returns 468 facial landmarks or None when no face is present."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
    ) -> None:
        self._mp = None
        self._mesh = None
        self._haar = None
        try:
            import mediapipe as mp

            self._mp = mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception:
            self._mesh = None

        # Fallback detector for low-light/noisy webcam feeds.
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                self._haar = None
        except Exception:
            self._haar = None

    def detect(self, frame) -> Optional[FaceDetection]:
        height, width = frame.shape[:2]
        if self._mesh is not None:
            rgb = frame[:, :, ::-1]
            results = self._mesh.process(rgb)
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                landmarks_norm = [[lm.x, lm.y, lm.z] for lm in face.landmark]
                landmarks_px = [
                    (int(max(0, min(width - 1, lm.x * width))), int(max(0, min(height - 1, lm.y * height))))
                    for lm in face.landmark
                ]

                xs = [p[0] for p in landmarks_px]
                ys = [p[1] for p in landmarks_px]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                return FaceDetection(
                    landmarks_norm=landmarks_norm,
                    landmarks_px=landmarks_px,
                    bbox_xyxy=(x1, y1, x2, y2),
                )

        # Fallback: use Haar for face presence and bbox only.
        if self._haar is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                return FaceDetection(landmarks_norm=[], landmarks_px=[], bbox_xyxy=(x1, y1, x2, y2))

        return None

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
