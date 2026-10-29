"""
Unit tests for DigitPredictor and model inference.
"""

import numpy as np
import pytest
from src.predict import DigitPredictor
from src.train import load_mnist_data


def test_digit_predictor_loading():
    predictor = DigitPredictor(model_path="models/digit_cnn.keras")
    assert predictor.is_model_loaded() is True


def test_digit_predictor_empty_image():
    predictor = DigitPredictor(model_path="models/digit_cnn.keras")
    blank = np.ones((200, 200, 3), dtype=np.uint8) * 255
    res = predictor.predict(blank)
    assert res["success"] is False
    assert "kosong" in res["error"].lower()


def test_digit_predictor_format_and_properties():
    predictor = DigitPredictor(model_path="models/digit_cnn.keras")
    _, _, x_test, y_test = load_mnist_data()

    # Take first test sample (MNIST image normalized [0, 1])
    sample = (x_test[0] * 255.0).astype(np.uint8).squeeze()
    res = predictor.predict(sample)

    assert res["success"] is True
    assert isinstance(res["digit"], int)
    assert 0 <= res["digit"] <= 9
    assert 0.0 <= res["confidence"] <= 100.0
    assert len(res["top_3"]) == 3
    assert len(res["probabilities"]) == 10

    # Top-3 should be descending
    assert res["top_3"][0]["confidence"] >= res["top_3"][1]["confidence"]
    assert res["top_3"][1]["confidence"] >= res["top_3"][2]["confidence"]

    # Sum of probabilities should be approximately 100%
    prob_sum = sum(res["probabilities"].values())
    assert 99.0 <= prob_sum <= 101.0

    # Check processed image shape
    assert res["processed_image"].shape == (28, 28)


def test_digit_predictor_accuracy_on_mnist_samples():
    predictor = DigitPredictor(model_path="models/digit_cnn.keras")
    _, _, x_test, y_test = load_mnist_data()

    # Test 10 consecutive MNIST test samples
    correct = 0
    for i in range(10):
        sample = (x_test[i] * 255.0).astype(np.uint8).squeeze()
        res = predictor.predict(sample)
        if res["digit"] == y_test[i]:
            correct += 1

    # Trained model should achieve at least 9/10 on random test batch
    assert correct >= 9


def test_math_predictor_loading():
    from src.predict import MathPredictor
    predictor = MathPredictor(model_path="models/math_cnn.keras")
    assert predictor.is_model_loaded() is True


def test_math_predictor_expression_evaluation():
    import cv2
    from src.predict import MathPredictor

    predictor = MathPredictor(model_path="models/math_cnn.keras")
    # Draw simple math expression: 2 strokes representing '+'
    canvas = np.ones((120, 240, 3), dtype=np.uint8) * 255
    # Digit 1 on left
    cv2.line(canvas, (40, 30), (40, 90), (0, 0, 0), thickness=6)
    # Operator + in middle
    cv2.line(canvas, (110, 60), (130, 60), (0, 0, 0), thickness=6)
    cv2.line(canvas, (120, 50), (120, 70), (0, 0, 0), thickness=6)
    # Digit 1 on right
    cv2.line(canvas, (190, 30), (190, 90), (0, 0, 0), thickness=6)

    res = predictor.predict_expression(canvas)
    assert res["success"] is True
    assert len(res["tokens"]) >= 2
    assert "calculation" in res

