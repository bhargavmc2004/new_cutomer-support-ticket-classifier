"""
Evaluate individual LLM classifiers without fallback.

This script imports the existing classification logic and synthetic TEST_DATASET,
temporarily overrides the MODEL_OPTIONS list so that each target model is tested
independently, and reports accuracy, precision, recall, and F1-score.

Usage:
    python evaluate_llm_accuracy.py
"""

import sys
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure src/ is on sys.path to import inference_api
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inference_api import MODEL_OPTIONS, classify_ticket_via_api  # noqa: E402
from evaluate_model import TEST_DATASET  # noqa: E402

TARGET_MODELS: List[str] = [
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


@contextmanager
def use_single_model(model_name: str):
    """
    Temporarily replace MODEL_OPTIONS with a single-model list so that
    classify_ticket_via_api runs without fallback.
    """
    original_models = list(MODEL_OPTIONS)
    try:
        MODEL_OPTIONS.clear()
        MODEL_OPTIONS.append(model_name)
        yield
    finally:
        MODEL_OPTIONS.clear()
        MODEL_OPTIONS.extend(original_models)


def evaluate_model(model_name: str) -> Dict[str, float]:
    """
    Run classification for all samples in TEST_DATASET using only the
    specified model and compute evaluation metrics.
    """
    y_true: List[str] = []
    y_pred: List[str] = []

    with use_single_model(model_name):
        for sample in TEST_DATASET:
            ticket = sample["ticket"]
            expected = sample["expected"]

            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()

            try:
                with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                    prediction = classify_ticket_via_api(ticket)
            except Exception:
                continue

            if not isinstance(prediction, str):
                continue

            normalized = prediction.strip().lower()
            if not normalized or normalized.startswith("[error]") or normalized == "invalid":
                continue

            y_true.append(expected)
            y_pred.append(normalized)

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_model_block(model_name: str, metrics: Dict[str, float]) -> None:
    """Print the per-model metrics block."""
    print("--------------------------------------")
    print(f"MODEL: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("--------------------------------------")


def print_summary_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print the final comparison table."""
    print("\nModel Name | Accuracy | Precision | Recall | F1")
    for model_name in TARGET_MODELS:
        metrics = results[model_name]
        print(
            f"{model_name} | "
            f"{metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f}"
        )


def main():
    results: Dict[str, Dict[str, float]] = {}

    for model_name in TARGET_MODELS:
        metrics = evaluate_model(model_name)
        results[model_name] = metrics
        print_model_block(model_name, metrics)

    print_summary_table(results)


if __name__ == "__main__":
    main()


