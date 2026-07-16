"""Frame renderer for UI overlays."""

from __future__ import annotations

import cv2

from display.color_palette import GREEN, ORANGE, RED, STATE_COLORS, WHITE
from display.components import draw_panel, draw_score_bar, draw_text
from display.layout import split_layout


class Renderer:
    def __init__(self, mode: str = "light") -> None:
        self.mode = mode

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _draw_detection(self, frame, detection, state_color) -> None:
        if detection is None:
            return

        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)

        for x, y in detection.landmarks_px:
            cv2.circle(frame, (x, y), 1, (80, 220, 80), -1)

    def render(
        self,
        frame,
        verdict_pkg,
        detection=None,
        warning: str | None = None,
        strict_mode: bool = False,
        popup_message: str | None = None,
        debug_overlay: bool = False,
        debug_lines: list[str] | None = None,
    ):
        h, w = frame.shape[:2]
        zones = split_layout(w, h)
        px, py, pw, ph = zones["panel"]

        draw_panel(frame, zones["panel"])

        if strict_mode:
            cv2.rectangle(frame, (12, 12), (180, 42), (20, 20, 20), -1)
            cv2.rectangle(frame, (12, 12), (180, 42), (0, 200, 255), 1)
            draw_text(frame, "STRICT MODE", 22, 33, color=(0, 220, 255), scale=0.55, thickness=1)

        state_color = STATE_COLORS.get(verdict_pkg.state, WHITE)
        self._draw_detection(frame, detection, state_color)
        draw_text(frame, f"Mode: {self.mode.upper()}", px + 16, py + 30, color=WHITE, scale=0.65, thickness=2)
        draw_text(frame, f"State: {verdict_pkg.state}", px + 16, py + 62, color=state_color, scale=0.65, thickness=2)
        draw_text(frame, f"Time: {verdict_pkg.time_remaining:0.1f}s", px + 16, py + 94)

        draw_score_bar(frame, "Deepfake", verdict_pkg.deepfake_score, px + 16, py + 130)
        draw_score_bar(frame, "Liveness", verdict_pkg.blink_score, px + 16, py + 180)

        risk = max(0.0, min(100.0, float(verdict_pkg.risk_score)))
        challenge_progress = getattr(verdict_pkg, "challenge_progress", "0/0")
        challenge_prompt = getattr(verdict_pkg, "challenge_prompt", None)
        show_challenge = challenge_progress not in ("off", "n/a") or bool(challenge_prompt)
        if show_challenge:
            draw_text(frame, f"Challenge: {challenge_progress}", px + 16, py + 238, scale=0.55, thickness=1)
            if challenge_prompt:
                draw_text(frame, f"Do now: {challenge_prompt}", px + 16, py + 260, color=(255, 230, 120), scale=0.55, thickness=1)
                risk_y = py + 286
            else:
                risk_y = py + 268
        else:
            risk_y = py + 246
        draw_text(frame, f"Risk Score: {risk:0.1f}", px + 16, risk_y, scale=0.7, thickness=2)

        if debug_overlay and debug_lines:
            dx = px + 16
            dy = py + 310
            for line in debug_lines[:6]:
                draw_text(frame, line, dx, dy, color=(200, 255, 255), scale=0.45, thickness=1)
                dy += 20

        if warning:
            cv2.rectangle(frame, (12, h - 46), (w - 12, h - 12), (0, 0, 180), -1)
            draw_text(frame, f"WARNING: {warning[:110]}", 20, h - 22, color=WHITE, scale=0.5, thickness=1)

        if popup_message:
            x1, y1, x2, y2 = 20, 56, min(w - 20, 700), 92
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            draw_text(frame, popup_message[:95], x1 + 10, y1 + 24, color=WHITE, scale=0.5, thickness=1)

        if verdict_pkg.verdict:
            verdict = verdict_pkg.verdict
            if verdict == "VERIFIED":
                color = GREEN
            elif verdict == "SUSPICIOUS":
                color = ORANGE
            else:
                color = RED

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            draw_text(frame, f"VERDICT: {verdict}", 40, 60, color=color, scale=1.1, thickness=3)

        return frame
