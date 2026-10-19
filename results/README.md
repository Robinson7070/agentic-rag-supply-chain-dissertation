# Results

Evaluation outputs used by the dashboard and the metrics scripts.

| File | What it is |
|---|---|
| `evaluation_results_20sets.json` | The system's verdict for each of the 20 synthetic document sets |
| `anomaly_log.json` | Anomalies flagged by the system, with severity and review status |
| `comparison_results.json` | The 6-case head-to-head run against the traditional RAG baseline |
| `metrics_20sets.json` | Set-level metrics, written by `python scripts/compute_metrics.py` |
| `statistical_results.json` | Output of `python src/statistical_significance.py` |

Recompute the set-level metrics at any time with `python scripts/compute_metrics.py`.
