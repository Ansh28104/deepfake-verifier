from __future__ import annotations

import pytest

from core.frame_extraction import extract_frames


def test_extract_frames_rejects_non_positive_frame_rate(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        extract_frames("dummy.mp4", str(tmp_path), frame_rate=0)
