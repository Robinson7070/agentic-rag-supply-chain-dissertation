"""
Statistical Significance Testing for Dissertation Evaluation
Victor Chukwudi Robinson — MSc AI & Data Science, UEL 2026

Computes:
1. Bootstrap confidence intervals on F1, Precision, Recall
2. McNemar's test - statistical comparison vs Traditional RAG baseline
3. Wilson score confidence intervals for proportions
4. Cohen's Kappa - inter-annotator agreement equivalent for system reliability

These make the evaluation publication-standard.
"""

import numpy as np
import json
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

# ============================================================
# GROUND TRUTH AND PREDICTIONS
# ============================================================

# 20-set evaluation results
# 1 = anomaly set, 0 = consistent/clean set
ground_truth = [
    0, 0, 0, 0, 0,   # Sets 01-05: clean
    1, 1, 1,         # Sets 06-08: price mismatch
    1, 1, 1,         # Sets 09-11: quantity mismatch
    1, 1, 1,         # Sets 12-14: missing document
    1, 1, 1,         # Sets 15-17: date inconsistency
    1, 1, 1,         # Sets 18-20: multiple errors
]

# Agentic RAG predictions (1 = predicted anomaly, 0 = predicted consistent)
agentic_predictions = [
    0, 0, 0, 0, 0,   # Sets 01-05: correctly CONSISTENT
    1, 1, 1,         # Sets 06-08: correctly ANOMALY
    1, 1, 1,         # Sets 09-11: correctly ANOMALY
    1, 1, 1,         # Sets 12-14: correctly ANOMALY
    1, 1, 1,         # Sets 15-17: correctly ANOMALY
    1, 1, 1,         # Sets 18-20: correctly ANOMALY
]

# Traditional RAG predictions (from comparison evaluation on 6 cases)
# Mapped to same 20 sets (partial evaluation)
# For McNemar's test we use the 6 directly compared cases
trad_ground_truth_6 = [1, 1, 1, 0, 0, 1]  # 6 test cases
trad_predictions_6  = [1, 1, 1, 0, 1, 0]  # Traditional RAG results
agent_predictions_6 = [1, 1, 1, 0, 0, 1]  # Agentic RAG results

ground_truth = np.array(ground_truth)
agentic_predictions = np.array(agentic_predictions)

# ============================================================
# BASIC METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    
    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
    }

# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for any metric."""
    n = len(y_true)
    bootstrap_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        bt_true = y_true[indices]
        bt_pred = y_pred[indices]
        score = metric_fn(bt_true, bt_pred)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_scores, alpha/2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    mean = np.mean(bootstrap_scores)
    
    return mean, lower, upper

def f1_fn(y_true, y_pred):
    return compute_metrics(y_true, y_pred)['f1']

def precision_fn(y_true, y_pred):
    return compute_metrics(y_true, y_pred)['precision']

def recall_fn(y_true, y_pred):
    return compute_metrics(y_true, y_pred)['recall']

def accuracy_fn(y_true, y_pred):
    return compute_metrics(y_true, y_pred)['accuracy']

# ============================================================
# WILSON SCORE CONFIDENCE INTERVALS
# ============================================================

def wilson_ci(count, nobs, ci=0.95):
    """Wilson score interval for proportions - better than normal approximation."""
    low, high = proportion_confint(count, nobs, alpha=1-ci, method='wilson')
    return low, high

# ============================================================
# MCNEMAR'S TEST
# ============================================================

def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test to compare two classifiers.
    Tests whether differences in performance are statistically significant.
    """
    # Build 2x2 contingency table
    # n00: both correct, n01: A wrong B right, n10: A right B wrong, n11: both wrong
    n00 = np.sum((pred_a == y_true) & (pred_b == y_true))
    n01 = np.sum((pred_a != y_true) & (pred_b == y_true))
    n10 = np.sum((pred_a == y_true) & (pred_b != y_true))
    n11 = np.sum((pred_a != y_true) & (pred_b != y_true))
    
    table = np.array([[n00, n01], [n10, n11]])
    
    if (n01 + n10) == 0:
        return None, 1.0, table  # No discordant pairs
    
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue, table

# ============================================================
# COHEN'S KAPPA
# ============================================================

def cohens_kappa(y_true, y_pred):
    """Cohen's Kappa - measures agreement beyond chance."""
    n = len(y_true)
    
    # Observed agreement
    po = np.sum(y_true == y_pred) / n
    
    # Expected agreement
    p1_true = np.sum(y_true == 1) / n
    p0_true = np.sum(y_true == 0) / n
    p1_pred = np.sum(y_pred == 1) / n
    p0_pred = np.sum(y_pred == 0) / n
    
    pe = (p1_true * p1_pred) + (p0_true * p0_pred)
    
    if pe == 1:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)
    return kappa

# ============================================================
# RUN ALL TESTS
# ============================================================

print("="*60)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("Agentic RAG System — Supply Chain Document Intelligence")
print("="*60)

# Basic metrics
metrics = compute_metrics(ground_truth, agentic_predictions)
print(f"\n1. BASIC METRICS (20 document sets)")
print(f"   True Positives:  {metrics['tp']}")
print(f"   True Negatives:  {metrics['tn']}")
print(f"   False Positives: {metrics['fp']}")
print(f"   False Negatives: {metrics['fn']}")
print(f"   Precision:       {metrics['precision']:.4f}")
print(f"   Recall:          {metrics['recall']:.4f}")
print(f"   F1 Score:        {metrics['f1']:.4f}")
print(f"   Accuracy:        {metrics['accuracy']:.4f}")

# Bootstrap CIs
print(f"\n2. BOOTSTRAP CONFIDENCE INTERVALS (n=10,000 iterations, 95% CI)")
f1_mean, f1_low, f1_high = bootstrap_ci(ground_truth, agentic_predictions, f1_fn)
prec_mean, prec_low, prec_high = bootstrap_ci(ground_truth, agentic_predictions, precision_fn)
rec_mean, rec_low, rec_high = bootstrap_ci(ground_truth, agentic_predictions, recall_fn)
acc_mean, acc_low, acc_high = bootstrap_ci(ground_truth, agentic_predictions, accuracy_fn)

print(f"   F1 Score:  {f1_mean:.4f} (95% CI: [{f1_low:.4f}, {f1_high:.4f}])")
print(f"   Precision: {prec_mean:.4f} (95% CI: [{prec_low:.4f}, {prec_high:.4f}])")
print(f"   Recall:    {rec_mean:.4f} (95% CI: [{rec_low:.4f}, {rec_high:.4f}])")
print(f"   Accuracy:  {acc_mean:.4f} (95% CI: [{acc_low:.4f}, {acc_high:.4f}])")

# Wilson score CIs
print(f"\n3. WILSON SCORE CONFIDENCE INTERVALS (95% CI)")
prec_w_low, prec_w_high = wilson_ci(metrics['tp'], metrics['tp'] + metrics['fp'])
rec_w_low, rec_w_high = wilson_ci(metrics['tp'], metrics['tp'] + metrics['fn'])
acc_w_low, acc_w_high = wilson_ci(metrics['tp'] + metrics['tn'], 20)

print(f"   Precision: [{prec_w_low:.4f}, {prec_w_high:.4f}]")
print(f"   Recall:    [{rec_w_low:.4f}, {rec_w_high:.4f}]")
print(f"   Accuracy:  [{acc_w_low:.4f}, {acc_w_high:.4f}]")

# McNemar's test
print(f"\n4. McNEMAR'S TEST (Agentic RAG vs Traditional RAG)")
y_true_6 = np.array(trad_ground_truth_6)
pred_agent_6 = np.array(agent_predictions_6)
pred_trad_6 = np.array(trad_predictions_6)

stat, pvalue, table = mcnemar_test(y_true_6, pred_agent_6, pred_trad_6)
print(f"   Contingency table:")
print(f"   Both correct:     {table[0][0]}")
print(f"   Only Trad correct:{table[0][1]}")
print(f"   Only Agent correct:{table[1][0]}")
print(f"   Both wrong:       {table[1][1]}")
if pvalue is not None:
    print(f"   Statistic: {stat}")
    print(f"   p-value:   {pvalue:.4f}")
    if pvalue < 0.05:
        print(f"   Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"   The agentic system significantly outperforms traditional RAG")
    else:
        print(f"   Result: p = {pvalue:.4f}")
        print(f"   Note: Small sample (n=6) limits statistical power")
        print(f"   Directional advantage confirmed: 6/6 vs 4/6 accuracy")
else:
    print("   No discordant pairs — systems agree on all cases")

# Cohen's Kappa
print(f"\n5. COHEN'S KAPPA")
kappa = cohens_kappa(ground_truth, agentic_predictions)
print(f"   Kappa: {kappa:.4f}")
if kappa >= 0.8:
    print(f"   Interpretation: Almost perfect agreement (κ ≥ 0.80)")
elif kappa >= 0.6:
    print(f"   Interpretation: Substantial agreement (κ ≥ 0.60)")
else:
    print(f"   Interpretation: Moderate agreement")

# Save results
results = {
    "basic_metrics": metrics,
    "bootstrap_ci": {
        "f1": {"mean": f1_mean, "ci_low": f1_low, "ci_high": f1_high},
        "precision": {"mean": prec_mean, "ci_low": prec_low, "ci_high": prec_high},
        "recall": {"mean": rec_mean, "ci_low": rec_low, "ci_high": rec_high},
        "accuracy": {"mean": acc_mean, "ci_low": acc_low, "ci_high": acc_high},
    },
    "wilson_ci": {
        "precision": {"low": prec_w_low, "high": prec_w_high},
        "recall": {"low": rec_w_low, "high": rec_w_high},
        "accuracy": {"low": acc_w_low, "high": acc_w_high},
    },
    "mcnemar": {
        "statistic": float(stat) if stat else None,
        "pvalue": float(pvalue) if pvalue else None,
        "contingency_table": table.tolist(),
        "significant": bool(pvalue < 0.05) if pvalue else False
    },
    "cohens_kappa": float(kappa),
    "n_samples": 20,
    "n_bootstrap": 10000,
}

with open("vector_store/statistical_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to vector_store/statistical_results.json")
print(f"\nFor dissertation reporting:")
print(f"  F1 = {f1_mean:.3f} (95% CI: [{f1_low:.3f}, {f1_high:.3f}])")
print(f"  Precision = {prec_mean:.3f} (95% CI: [{prec_low:.3f}, {prec_high:.3f}])")
print(f"  Recall = {rec_mean:.3f} (95% CI: [{rec_low:.3f}, {rec_high:.3f}])")
print(f"  Cohen's Kappa = {kappa:.3f} (almost perfect agreement)")
