"""
Model Evaluation Module for Handwritten Digit CNN.

Computes:
- Test Accuracy, Loss
- Precision, Recall, F1-score (per-class & macro/weighted)
- Confusion Matrix plot (saved to reports/confusion_matrix.png)
- Classification report (saved to reports/classification_report.txt)
- Sample correct and misclassified predictions for error analysis
"""

import json
import os
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from src.train import load_mnist_data, DIGIT_CLASSES


def evaluate_digit_model(
    model_path: str = "models/digit_cnn.keras",
    report_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Evaluate the saved digit CNN model on MNIST test dataset.
    Generates reports/confusion_matrix.png and reports/classification_report.txt.
    Returns metrics dictionary for Streamlit evaluation dashboard.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    os.makedirs(report_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    _, _, x_test, y_test = load_mnist_data()

    print("Evaluating on 10,000 MNIST test images...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Predictions
    raw_predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(raw_predictions, axis=1)

    # Classification Report
    clf_report_str = classification_report(
        y_test,
        y_pred,
        target_names=DIGIT_CLASSES,
        digits=4,
    )
    clf_report_dict = classification_report(
        y_test,
        y_pred,
        target_names=DIGIT_CLASSES,
        output_dict=True,
    )

    # Save text report
    report_txt_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"=== MNIST Digit Recognition CNN Evaluation Report ===\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(clf_report_str)
    print(f"Classification report saved to {report_txt_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cax = ax.matshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Annotate numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(DIGIT_CLASSES, fontsize=10)
    ax.set_yticklabels(DIGIT_CLASSES, fontsize=10)
    ax.set_xlabel("Predicted Digit", fontsize=11, labelpad=10)
    ax.set_ylabel("True Digit", fontsize=11)
    ax.set_title(f"Confusion Matrix (Test Accuracy: {test_acc*100:.2f}%)", fontsize=13, pad=15)
    plt.tight_layout()

    cm_png_path = os.path.join(report_dir, "confusion_matrix.png")
    fig.savefig(cm_png_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {cm_png_path}")

    # Sample correct & misclassified items for visualization
    correct_indices = np.where(y_pred == y_test)[0]
    misclassified_indices = np.where(y_pred != y_test)[0]

    sample_correct = []
    for idx in correct_indices[:6]:
        sample_correct.append({
            "index": int(idx),
            "true": int(y_test[idx]),
            "pred": int(y_pred[idx]),
            "confidence": float(raw_predictions[idx][y_pred[idx]]),
            "image": x_test[idx].squeeze().tolist(),
        })

    sample_misclassified = []
    for idx in misclassified_indices[:6]:
        sample_misclassified.append({
            "index": int(idx),
            "true": int(y_test[idx]),
            "pred": int(y_pred[idx]),
            "confidence": float(raw_predictions[idx][y_pred[idx]]),
            "image": x_test[idx].squeeze().tolist(),
        })

    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "precision_macro": float(clf_report_dict["macro avg"]["precision"]),
        "recall_macro": float(clf_report_dict["macro avg"]["recall"]),
        "f1_macro": float(clf_report_dict["macro avg"]["f1-score"]),
        "precision_weighted": float(clf_report_dict["weighted avg"]["precision"]),
        "recall_weighted": float(clf_report_dict["weighted avg"]["recall"]),
        "f1_weighted": float(clf_report_dict["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cls: {
                "precision": float(clf_report_dict[cls]["precision"]),
                "recall": float(clf_report_dict[cls]["recall"]),
                "f1": float(clf_report_dict[cls]["f1-score"]),
                "support": int(clf_report_dict[cls]["support"]),
            }
            for cls in DIGIT_CLASSES
        },
        "sample_correct": sample_correct,
        "sample_misclassified": sample_misclassified,
    }

    # Save metrics JSON for quick loading in Streamlit
    metrics_json_path = os.path.join(report_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate_digit_model()
