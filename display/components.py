"""Reusable drawing components."""

from __future__ import annotations

import cv2

from display.color_palette import GRAY, PANEL_BG, WHITE


def draw_panel(frame, panel_rect):
    x, y, w, h = panel_rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), PANEL_BG, -1)


def draw_text(frame, text: str, x: int, y: int, color=WHITE, scale: float = 0.6, thickness: int = 1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_score_bar(frame, label: str, value: float, x: int, y: int, w: int = 220, h: int = 18):
    value = max(0.0, min(1.0, float(value)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)
    fill_w = int(w * value)
    color = (0, int(255 * (1 - value)), int(255 * value))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    draw_text(frame, f"{label}: {int(value * 100)}%", x, y - 6)
