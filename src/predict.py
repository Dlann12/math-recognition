"""
Inference module for Digit Recognition and Math Expression Recognition.

Provides:
- DigitPredictor: predicts single handwritten digit (0-9) with confidence & top-3
- MathPredictor: segments handwritten math expressions, recognizes symbols, and calculates results
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.math_parser import evaluate_expression
from src.preprocess import preprocess_single_image, segment_math_expression
from src.train import DIGIT_CLASSES, MATH_CLASSES


class DigitPredictor:
    """Predictor for single handwritten digit classification."""

    def __init__(self, model_path: str = "models/digit_cnn.keras"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = None

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_input) -> Dict[str, Any]:
        """
        Run inference on a single digit canvas or uploaded image.
        Returns:
            dict containing:
                - success: bool
                - digit: int (0-9)
                - confidence: float (0.0 to 100.0)
                - top_3: list of dicts [{"digit": d, "confidence": c}]
                - probabilities: dict of {digit_str: float_percent}
                - processed_image: 28x28 numpy array
                - error: str or None
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                return {
                    "success": False,
                    "error": f"Model belum ditemukan di '{self.model_path}'. Silakan jalankan proses training terlebih dahulu.",
                }

        processed_tensor, is_valid, msg = preprocess_single_image(image_input)
        if not is_valid or processed_tensor is None:
            return {
                "success": False,
                "error": msg,
            }

        preds = self.model.predict(processed_tensor, verbose=0)[0]
        predicted_idx = int(np.argmax(preds))
        confidence = float(preds[predicted_idx] * 100.0)

        # Top 3 predictions
        top_indices = np.argsort(preds)[::-1][:3]
        top_3 = [
            {"digit": int(idx), "confidence": round(float(preds[idx] * 100.0), 2)}
            for idx in top_indices
        ]

        probabilities = {
            str(i): round(float(preds[i] * 100.0), 2)
            for i in range(10)
        }

        return {
            "success": True,
            "digit": predicted_idx,
            "confidence": round(confidence, 2),
            "top_3": top_3,
            "probabilities": probabilities,
            "processed_image": processed_tensor[0, :, :, 0],
            "error": None,
        }


class MathPredictor:
    """Predictor for multi-character handwritten math expression recognition."""

    def __init__(self, model_path: str = "models/math_cnn.keras"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = None

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def predict_expression(self, image_input) -> Dict[str, Any]:
        """
        Segment canvas/image into characters, classify each, assemble equation, and calculate result.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                return {
                    "success": False,
                    "error": f"Model matematika belum ditemukan di '{self.model_path}'. Silakan jalankan training model matematika terlebih dahulu.",
                }

        segments = segment_math_expression(image_input)
        if not segments:
            return {
                "success": False,
                "error": "Tidak ada simbol atau ekspresi matematika yang terdeteksi pada kanvas/gambar.",
            }

        tokens = []
        expression_chars = []

        for tensor, box in segments:
            preds = self.model.predict(tensor, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            char = MATH_CLASSES[pred_idx]
            conf = float(preds[pred_idx] * 100.0)

            # Heuristic geometric sanity check for ambiguous characters:
            # - '-' is very flat (w > 2.2 * h)
            # - '=' has two horizontal components
            bx, by, bw, bh = box
            aspect_ratio = float(bw) / float(max(1, bh))
            if aspect_ratio > 2.5 and bh < 25 and char not in ["-", "="]:
                char = "-"
                conf = max(conf, 85.0)

            tokens.append({
                "char": char,
                "confidence": round(conf, 2),
                "box": box,
                "tensor_image": tensor[0, :, :, 0],
            })
            expression_chars.append(char)

        reconstructed_str = "".join(expression_chars)

        # Safely evaluate with math_parser
        calc_result = evaluate_expression(reconstructed_str)

        return {
            "success": True,
            "expression": reconstructed_str,
            "tokens": tokens,
            "calculation": calc_result,
            "error": None,
        }
