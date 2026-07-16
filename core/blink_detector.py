"""Blink and liveness signal extraction from face landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Optional, Sequence


def _eye_aspect_ratio(eye_points: Sequence[Sequence[float]]) -> float:
	"""Compute EAR from six eye points in the standard order.

	EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
	"""
	p1, p2, p3, p4, p5, p6 = eye_points
	vertical = dist(p2, p6) + dist(p3, p5)
	horizontal = 2.0 * dist(p1, p4)
	if horizontal == 0:
		return 0.0
	return vertical / horizontal


@dataclass
class BlinkMetrics:
	ear: float
	blink_count: int
	blink_pattern_score: float
	passive_motion_score: float


class BlinkDetector:
	"""Tracks blink count and returns a normalized liveness score."""

	# MediaPipe Face Mesh eye indices (approximate and widely used)
	LEFT_EYE = (33, 160, 158, 133, 153, 144)
	RIGHT_EYE = (362, 385, 387, 263, 373, 380)

	def __init__(self, ear_threshold: float = 0.20, consecutive_frames: int = 2) -> None:
		self.ear_threshold = ear_threshold
		self.consecutive_frames = consecutive_frames
		self._closed_counter = 0
		self._blink_count = 0
		self._ear_history: list[float] = []

	def _extract_eye(self, landmarks: Sequence[Sequence[float]], indices: Sequence[int]) -> list[list[float]]:
		return [[float(landmarks[i][0]), float(landmarks[i][1])] for i in indices]

	def process(self, landmarks: Optional[Sequence[Sequence[float]]]) -> BlinkMetrics:
		"""Compute EAR, update blink count, and return liveness score in [0, 1]."""
		if landmarks is None or len(landmarks) < 388:
			return BlinkMetrics(
				ear=0.0,
				blink_count=self._blink_count,
				blink_pattern_score=0.5,
				passive_motion_score=0.5,
			)

		left = self._extract_eye(landmarks, self.LEFT_EYE)
		right = self._extract_eye(landmarks, self.RIGHT_EYE)
		ear = (_eye_aspect_ratio(left) + _eye_aspect_ratio(right)) / 2.0

		if ear < self.ear_threshold:
			self._closed_counter += 1
		else:
			if self._closed_counter >= self.consecutive_frames:
				self._blink_count += 1
			self._closed_counter = 0

		self._ear_history.append(ear)
		if len(self._ear_history) > 20:
			self._ear_history.pop(0)

		# Higher is more suspicious: closed eyes too often or unnaturally still.
		if ear < self.ear_threshold * 0.8:
			blink_pattern_score = 0.8
		elif ear < self.ear_threshold:
			blink_pattern_score = 0.6
		else:
			blink_pattern_score = 0.2

		ear_range = 0.0
		if len(self._ear_history) >= 3:
			ear_range = max(self._ear_history) - min(self._ear_history)

		if ear_range < 0.015:
			passive_motion_score = 0.75
		elif ear_range < 0.03:
			passive_motion_score = 0.5
		else:
			passive_motion_score = 0.25

		return BlinkMetrics(
			ear=ear,
			blink_count=self._blink_count,
			blink_pattern_score=blink_pattern_score,
			passive_motion_score=passive_motion_score,
		)
