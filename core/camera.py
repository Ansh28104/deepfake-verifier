"""Camera capture module."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2


class CameraStream:
	"""Reads frames from a webcam and handles transient capture failures."""

	def __init__(self, camera_index: int = 0, unmirror: bool = False) -> None:
		self.cap = cv2.VideoCapture(camera_index)
		# Best-effort low-latency capture settings for live inference.
		set_prop = getattr(self.cap, "set", None)
		if callable(set_prop):
			set_prop(cv2.CAP_PROP_BUFFERSIZE, 1)
			set_prop(cv2.CAP_PROP_FRAME_WIDTH, 640)
			set_prop(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self.unmirror = unmirror
		self._last_frame = None
		self._failures = 0

	def is_opened(self) -> bool:
		opened = getattr(self.cap, "isOpened", None)
		if callable(opened):
			return bool(opened())
		return True

	def read(self) -> Tuple[bool, Optional[object]]:
		ok, frame = self.cap.read()
		if ok:
			if self.unmirror:
				frame = cv2.flip(frame, 1)
			self._last_frame = frame
			self._failures = 0
			return True, frame

		self._failures += 1
		return False, None

	def too_many_failures(self, max_failures: int = 10) -> bool:
		return self._failures >= max_failures

	def release(self) -> None:
		if self.cap is not None:
			self.cap.release()
