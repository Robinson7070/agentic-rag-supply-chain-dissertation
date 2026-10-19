"""Set-level evaluation metrics, computed from the committed results file.

Ground truth for the 20-set synthetic corpus is defined by purchase-order reference
(see ground_truth/MASTER_GROUND_TRUTH.txt):

  * PO-2026-1001 .. PO-2026-1005  -> the five consistent sets (no anomaly)
  * every other set               -> contains at least one anomaly. This includes
                                     PO-2026-2001 .. PO-2026-2014 and the Set 14 case
                                     with no purchase order, reported as NOT_PROVIDED.

A set is "predicted anomalous" when the system's result has consistent == False.
"""

import json
import re

CLEAN_PO_PATTERN = re.compile(r"PO-2026-1\d{3}")


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_has_anomaly(po_ref):
    """Ground truth label for a set, from its purchase-order reference."""
    return CLEAN_PO_PATTERN.fullmatch(str(po_ref)) is None


def labels_and_predictions(eval_results):
    """Return (ground_truth, predictions) lists where 1 = anomaly set, 0 = consistent set."""
    truth = [1 if expected_has_anomaly(r["po_ref"]) else 0 for r in eval_results]
    pred = [0 if r["consistent"] else 1 for r in eval_results]
    return truth, pred


def binary_metrics(eval_results):
    """Confusion matrix and precision / recall / F1 / accuracy at set level."""
    truth, pred = labels_and_predictions(eval_results)
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    n = len(truth)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def misclassified(eval_results):
    """po_ref of every set where the system's verdict disagrees with the ground truth."""
    out = []
    for r in eval_results:
        expected = expected_has_anomaly(r["po_ref"])
        predicted = not r["consistent"]
        if expected != predicted:
            out.append((r["po_ref"], "expected anomaly" if expected else "expected consistent"))
    return out
