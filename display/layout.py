"""Layout helpers for display panel zones."""


def split_layout(width: int, height: int) -> dict:
    panel_width = int(width * 0.30)
    return {
        "frame": (0, 0, width - panel_width, height),
        "panel": (width - panel_width, 0, panel_width, height),
    }
