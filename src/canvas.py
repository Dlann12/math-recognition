"""
Drawing Canvas Utilities and Visual Overlay Helpers.

Provides configuration presets for streamlit-drawable-canvas and helper functions
to overlay detected characters and bounding boxes onto images.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image


def get_canvas_config(mode: str = "digit") -> Dict[str, Any]:
    """
    Return default configuration for streamlit-drawable-canvas.
    mode: 'digit' (square, large stroke) or 'math' (wide, medium stroke)
    """
    if mode == "digit":
        return {
            "fill_color": "rgba(255, 165, 0, 0.0)",
            "stroke_width": 14,
            "stroke_color": "#000000",
            "background_color": "#FFFFFF",
            "height": 280,
            "width": 280,
            "drawing_mode": "freedraw",
            "return_image_data": True,
            "key": "digit_canvas",
        }
    else:
        return {
            "fill_color": "rgba(255, 165, 0, 0.0)",
            "stroke_width": 10,
            "stroke_color": "#000000",
            "background_color": "#FFFFFF",
            "height": 240,
            "width": 560,
            "drawing_mode": "freedraw",
            "return_image_data": True,
            "key": "math_canvas",
        }


def draw_bounding_boxes(
    image_array: np.ndarray,
    tokens: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (37, 99, 235),
    text_color: Tuple[int, int, int] = (220, 38, 38),
) -> np.ndarray:
    """
    Draw bounding boxes and detected character labels on the canvas image.
    Returns RGB image suitable for st.image.
    """
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        rgb = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2BGR)
    elif len(image_array.shape) == 3:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    annotated = rgb.copy()

    for idx, token in enumerate(tokens):
        bx, by, bw, bh = token["box"]
        char = token["char"]
        conf = token["confidence"]

        # Draw box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 2)

        # Draw label header
        label = f"{char} ({conf:.0f}%)"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Label background
        label_y1 = max(0, by - h_text - 6)
        label_y2 = by
        label_x1 = bx
        label_x2 = min(annotated.shape[1], bx + w_text + 6)
        cv2.rectangle(annotated, (label_x1, label_y1), (label_x2, label_y2), box_color, -1)

        # Label text
        cv2.putText(
            annotated,
            label,
            (bx + 3, by - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
