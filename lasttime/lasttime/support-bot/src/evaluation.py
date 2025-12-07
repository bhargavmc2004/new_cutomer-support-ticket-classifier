"""
Minimal evaluation script placed in src/ to compute and print final model metrics.

This mirrors the behavior required: no per-sample logs, no CSV saving, only final metrics block.
"""

from contextlib import redirect_stdout, redirect_stderr
import io
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from inference_api import classify_ticket_via_api, get_model_name

# NOTE: The TEST_DATASET is expected to be provided by the project-level evaluation script.
# For convenience we import it if available; otherwise callers can supply their own.
try:
    from ..evaluate_model import TEST_DATASET  # noqa: F401
except Exception:
    # If TEST_DATASET isn't available via relative import, define an empty placeholder.
    TEST_DATASET = []


def run_evaluation(dataset=None):
    data = dataset if dataset is not None else TEST_DATASET

    y_true = []
    y_pred = []

    for sample in data:
        ticket = sample.get("ticket")
        expected = sample.get("expected")

        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        try:
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                predicted = classify_ticket_via_api(ticket)
        except Exception:
            continue

        if isinstance(predicted, str) and predicted.startswith("[Error]"):
            continue

        try:
            pred_norm = predicted.lower().strip()
        except Exception:
            continue

        y_true.append(expected)
        y_pred.append(pred_norm)

    if len(y_true) == 0:
        total = 0
        accuracy = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    else:
        total = len(y_true)
        accuracy = accuracy_score(y_true, y_pred) * 100.0
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

    model_name = get_model_name() if callable(get_model_name) else str(get_model_name)
    print("============================================================")
    print("                  FINAL MODEL METRICS")
    print("============================================================")
    print(f"Model Used: {model_name}")
    print(f"Total Samples Evaluated: {total}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (Weighted): {weighted_precision:.2f}")
    print(f"Recall (Weighted): {weighted_recall:.2f}")
    print(f"F1-Score (Weighted): {weighted_f1:.2f}")
    print("============================================================")


if __name__ == '__main__':
    run_evaluation()
