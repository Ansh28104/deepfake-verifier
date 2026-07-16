"""Color constants used by the renderer."""

PANEL_BG = (25, 25, 25)
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
GREEN = (70, 200, 90)
ORANGE = (0, 165, 255)
RED = (40, 40, 220)
BLUE = (220, 140, 20)
YELLOW = (0, 220, 220)

STATE_COLORS = {
    "WAITING": WHITE,
    "FACE_DETECTED": BLUE,
    "ANALYZING": YELLOW,
    "VERDICT": WHITE,
    "RESET": GRAY,
}
