"""
Training pipeline for Handwritten Digit CNN and Math Expression CNN.

Implements PRD Section 8:
- Baseline Digit CNN (Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(128) -> Dropout(0.3) -> Dense(10))
- 15-class Math CNN (Digits 0-9 + Operators +, -, *, /, =)
- Early stopping, validation split, model checkpointing in .keras format.
"""

import argparse
import os
import random
from typing import Tuple
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocess import normalize_to_mnist_format

# Class definitions
DIGIT_CLASSES = [str(i) for i in range(10)]
MATH_CLASSES = [str(i) for i in range(10)] + ["+", "-", "*", "/", "="]


def build_digit_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 10) -> keras.Model:
    """
    Construct CNN model following PRD Section 8 specifications:
    1. Input: 28 x 28 x 1
    2. Conv2D: 32 filters, 3x3, ReLU
    3. MaxPooling2D: 2x2
    4. Conv2D: 64 filters, 3x3, ReLU
    5. MaxPooling2D: 2x2
    6. Flatten
    7. Dense: 128 neurons, ReLU
    8. Dropout: 0.3
    9. Dense: num_classes, Softmax
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="input_image"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv2d_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_features"),
            layers.Dropout(0.3, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="output_probabilities"),
        ],
        name="HandwrittenDigitCNN" if num_classes == 10 else "HandwrittenMathCNN",
    )
    return model


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and normalize MNIST dataset.
    Returns:
        (x_train, y_train), (x_test, y_test)
        x arrays are shape (N, 28, 28, 1), normalized to [0, 1] float32
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
    x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]

    return x_train, y_train, x_test, y_test


def generate_operator_samples(operator_char: str, count: int = 1500) -> np.ndarray:
    """
    Generate synthetic augmented handwritten-style operator images for math dataset.
    """
    samples = []
    size = (56, 56)  # draw at high res then downscale & center

    for _ in range(count):
        img = np.zeros(size, dtype=np.uint8)
        thickness = random.choice([2, 3, 4, 5])
        angle = random.uniform(-15, 15)

        if operator_char == "+":
            cx, cy = 28 + random.randint(-3, 3), 28 + random.randint(-3, 3)
            length = random.randint(14, 22)
            cv2.line(img, (cx - length, cy), (cx + length, cy), 255, thickness)
            cv2.line(img, (cx, cy - length), (cx, cy + length), 255, thickness)

        elif operator_char == "-":
            cx, cy = 28 + random.randint(-2, 2), 28 + random.randint(-3, 3)
            length = random.randint(14, 24)
            cv2.line(img, (cx - length, cy), (cx + length, cy), 255, thickness)

        elif operator_char == "*":
            cx, cy = 28 + random.randint(-2, 2), 28 + random.randint(-2, 2)
            length = random.randint(10, 16)
            is_cross = random.random() < 0.5
            if is_cross:
                cv2.line(img, (cx - length, cy - length), (cx + length, cy + length), 255, thickness)
                cv2.line(img, (cx - length, cy + length), (cx + length, cy - length), 255, thickness)
            else:
                cv2.line(img, (cx - length, cy), (cx + length, cy), 255, thickness)
                cv2.line(img, (cx, cy - length), (cx, cy + length), 255, thickness)
                cv2.line(img, (cx - length, cy - length), (cx + length, cy + length), 255, thickness)
                cv2.line(img, (cx - length, cy + length), (cx + length, cy - length), 255, thickness)

        elif operator_char == "/":
            cx, cy = 28 + random.randint(-2, 2), 28 + random.randint(-2, 2)
            dx = random.randint(10, 16)
            dy = random.randint(14, 22)
            cv2.line(img, (cx - dx, cy + dy), (cx + dx, cy - dy), 255, thickness)

        elif operator_char == "=":
            cx, cy = 28 + random.randint(-2, 2), 28 + random.randint(-2, 2)
            length = random.randint(14, 22)
            gap = random.randint(5, 9)
            cv2.line(img, (cx - length, cy - gap), (cx + length, cy - gap), 255, thickness)
            cv2.line(img, (cx - length, cy + gap), (cx + length, cy + gap), 255, thickness)

        # Apply rotation
        M = cv2.getRotationMatrix2D((28, 28), angle, 1.0)
        rotated = cv2.warpAffine(img, M, size, borderValue=0)

        # Slight blur for smooth edges
        blurred = cv2.GaussianBlur(rotated, (3, 3), 0)

        # Pass through MNIST normalizer to match 28x28 center of mass standard
        norm_tensor, is_valid = normalize_to_mnist_format(blurred)
        if is_valid:
            samples.append(norm_tensor[0])

    return np.array(samples, dtype=np.float32)


def train_digit_model(epochs: int = 10, batch_size: int = 64, save_path: str = "models/digit_cnn.keras"):
    """
    Train and save baseline 10-class digit CNN model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("=== Loading MNIST Dataset for Digit Model ===")
    x_train, y_train, x_test, y_test = load_mnist_data()

    print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

    # Split train into train & validation (50k train, 10k val)
    x_val = x_train[50000:]
    y_val = y_train[50000:]
    x_train = x_train[:50000]
    y_train = y_train[:50000]

    model = build_digit_cnn(num_classes=10)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n=== Training Digit CNN ===")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    print("\n=== Evaluating Digit CNN on MNIST Test Set ===")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    model.save(save_path)
    print(f"Model successfully saved to {save_path}")
    return model, history


def train_math_model(epochs: int = 10, batch_size: int = 64, save_path: str = "models/math_cnn.keras"):
    """
    Train and save 15-class Math CNN model (digits 0-9 and operators +, -, *, /, =).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("=== Preparing Dataset for Math Model (15 Classes) ===")
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_mnist_data()

    # Sample balanced subset of digits (e.g. 2000 per digit)
    math_x_train = []
    math_y_train = []
    math_x_test = []
    math_y_test = []

    for digit in range(10):
        train_idx = np.where(y_train_mnist == digit)[0][:2500]
        test_idx = np.where(y_test_mnist == digit)[0][:500]

        math_x_train.append(x_train_mnist[train_idx])
        math_y_train.append(y_train_mnist[train_idx])

        math_x_test.append(x_test_mnist[test_idx])
        math_y_test.append(y_test_mnist[test_idx])

    # Generate synthetic samples for operators: +, -, *, /, =
    operators = ["+", "-", "*", "/", "="]
    for idx, op in enumerate(operators):
        class_id = 10 + idx
        print(f"Generating synthetic operator samples for '{op}' (class {class_id})...")
        op_samples = generate_operator_samples(op, count=2500)
        train_samples = op_samples[:2000]
        test_samples = op_samples[2000:2500]

        math_x_train.append(train_samples)
        math_y_train.append(np.full(len(train_samples), class_id, dtype=np.int32))

        math_x_test.append(test_samples)
        math_y_test.append(np.full(len(test_samples), class_id, dtype=np.int32))

    X_train = np.vstack(math_x_train)
    Y_train = np.concatenate(math_y_train)
    X_test = np.vstack(math_x_test)
    Y_test = np.concatenate(math_y_test)

    # Shuffle training set
    perm_train = np.random.permutation(len(X_train))
    X_train = X_train[perm_train]
    Y_train = Y_train[perm_train]

    perm_test = np.random.permutation(len(X_test))
    X_test = X_test[perm_test]
    Y_test = Y_test[perm_test]

    print(f"Total Math Dataset: {len(X_train)} train, {len(X_test)} test, 15 classes")

    model = build_digit_cnn(num_classes=15)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n=== Training Math CNN ===")
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n=== Evaluating Math CNN ===")
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Math Test Loss: {test_loss:.4f} | Math Test Accuracy: {test_acc * 100:.2f}%")

    model.save(save_path)
    print(f"Math Model successfully saved to {save_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN models for digit and math recognition.")
    parser.add_argument("--model-type", type=str, choices=["digit", "math", "all"], default="all",
                        help="Which model to train: digit, math, or all")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()

    if args.model_type in ["digit", "all"]:
        train_digit_model(epochs=args.epochs, batch_size=args.batch_size)

    if args.model_type in ["math", "all"]:
        train_math_model(epochs=args.epochs, batch_size=args.batch_size)
