"""
Unit tests for preprocessing and segmentation.
"""

import numpy as np
import cv2
from src.preprocess import (
    is_image_empty,
    normalize_to_mnist_format,
    preprocess_single_image,
    segment_math_expression,
)


def test_is_image_empty():
    # Completely white canvas
    white_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    assert is_image_empty(white_img) is True

    # Completely black canvas
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    assert is_image_empty(black_img) is True

    # Canvas with drawn stroke
    drawn_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.line(drawn_img, (50, 50), (150, 150), (0, 0, 0), thickness=8)
    assert is_image_empty(drawn_img) is False


def test_normalize_to_mnist_format():
    # Create an image with a drawn digit '1'
    img = np.ones((100, 100), dtype=np.uint8) * 255
    cv2.line(img, (50, 20), (50, 80), 0, thickness=6)

    tensor, is_valid = normalize_to_mnist_format(img)
    assert is_valid is True
    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    # There should be non-zero elements
    assert np.sum(tensor) > 0


def test_preprocess_single_image_empty():
    blank = np.ones((150, 150, 3), dtype=np.uint8) * 255
    tensor, is_valid, msg = preprocess_single_image(blank)
    assert is_valid is False
    assert tensor is None
    assert "kosong" in msg.lower()


def test_preprocess_single_image_valid():
    # Canvas with a circle (digit '0')
    img = np.ones((150, 150, 3), dtype=np.uint8) * 255
    cv2.circle(img, (75, 75), 30, (0, 0, 0), thickness=6)

    tensor, is_valid, msg = preprocess_single_image(img)
    assert is_valid is True
    assert tensor.shape == (1, 28, 28, 1)
    assert "berhasil" in msg.lower()


def test_segment_math_expression():
    # Create image with two separated digits: '1' and '2' side by side
    canvas = np.ones((100, 250, 3), dtype=np.uint8) * 255
    # First character at x=40
    cv2.line(canvas, (40, 20), (40, 80), (0, 0, 0), thickness=6)
    # Second character at x=140
    cv2.line(canvas, (140, 20), (180, 20), (0, 0, 0), thickness=6)
    cv2.line(canvas, (180, 20), (140, 80), (0, 0, 0), thickness=6)
    cv2.line(canvas, (140, 80), (180, 80), (0, 0, 0), thickness=6)

    segments = segment_math_expression(canvas)
    assert len(segments) == 2
    # Verify sorted left-to-right
    x1 = segments[0][1][0]
    x2 = segments[1][1][0]
    assert x1 < x2
    assert segments[0][0].shape == (1, 28, 28, 1)
    assert segments[1][0].shape == (1, 28, 28, 1)
