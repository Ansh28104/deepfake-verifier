from __future__ import annotations

import numpy as np

from core.camera import CameraStream


class _FakeCapture:
    def __init__(self, results):
        self._results = list(results)
        self._index = 0
        self.released = False

    def read(self):
        if self._index >= len(self._results):
            return False, None
        value = self._results[self._index]
        self._index += 1
        return value

    def release(self):
        self.released = True


def test_camera_read_failure_does_not_return_stale_frame(monkeypatch) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    fake_capture = _FakeCapture([(True, frame), (False, None)])

    monkeypatch.setattr("core.camera.cv2.VideoCapture", lambda *_: fake_capture)

    camera = CameraStream(camera_index=0)

    ok, first = camera.read()
    assert ok is True
    assert first is frame

    ok, second = camera.read()
    assert ok is False
    assert second is None
    assert camera.too_many_failures(max_failures=1) is True

    camera.release()
    assert fake_capture.released is True
