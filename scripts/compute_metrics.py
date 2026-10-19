"""Recompute set-level metrics from the committed evaluation results.

Usage (from the repository root):
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py path/to/evaluation_results.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from metrics_utils import binary_metrics, load_results, misclassified  # noqa: E402

DEFAULT_PATH = os.path.join("results", "evaluation_results_20sets.json")
OUT_PATH = os.path.join("results", "metrics_20sets.json")

path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
results = load_results(path)
m = binary_metrics(results)

print(f"Results file: {path}")
print(f"Sets evaluated: {m['n']}")
print(f"Confusion matrix: TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
print(f"Precision: {m['precision']:.3f}")
print(f"Recall:    {m['recall']:.3f}")
print(f"F1:        {m['f1']:.3f}")
print(f"Accuracy:  {m['accuracy']:.3f}")

wrong = misclassified(results)
if wrong:
    print("\nSets where the verdict disagrees with the ground truth:")
    for po, why in wrong:
        print(f"  {po}: {why}")
else:
    print("\nEvery set matches the ground truth.")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(m, f, indent=2)
print(f"\nSaved to {OUT_PATH}")
