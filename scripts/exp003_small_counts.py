"""Experiment 3: three treatments of small total_count rows (decision D4), D5 split, seed 42.

Reference: mean of invalid_ratio by cell of size 1e-6 (experiment 2).
Each variant changes one thing.
Run from the repository root: PYTHONPATH=src python scripts/exp003_small_counts.py [group ...]
Groups: reference, A, B, C, Cw (default: all). Run them one by one on a small machine.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from parking.baselines import (
    group_mean_predict,
    score_predictions,
    shrunk_group_mean_predict,
    to_markdown,
)
from parking.data import load_train
from parking.zones import assign_cells

FINE, PARENT = 1e-6, 1e-5
MIN_COUNTS = [2, 6, 45]
SHRINK_STRENGTHS = [0, 1, 3, 10, 30]
WEIGHTED_STRENGTHS = [3, 10, 30]


def main() -> None:
    X, y = load_train(Path("data/raw"))
    fit, val = X["month_of_year"] <= 9, X["month_of_year"] >= 10
    X_fit, y_fit, X_val, y_val = X[fit], y[fit], X[val], y[val]
    total_fit, total_val = X_fit["total_count"], X_val["total_count"]
    fine_fit, fine_val = assign_cells(X_fit, FINE), assign_cells(X_val, FINE)
    parent_fit, parent_val = assign_cells(X_fit, PARENT), assign_cells(X_val, PARENT)

    groups = set(sys.argv[1:]) or {"reference", "A", "B", "C", "Cw"}
    predictions = {}
    if "reference" in groups:
        predictions["reference"] = group_mean_predict(fine_fit, y_fit, fine_val)
    for minimum in MIN_COUNTS if "A" in groups else []:  # option A: filter
        keep = (total_fit >= minimum).to_numpy()
        predictions[f"A_filter_min{minimum}"] = group_mean_predict(
            fine_fit[keep], y_fit[keep], fine_val
        )
    weightings = {"sqrt": np.sqrt(total_fit), "linear": total_fit} if "B" in groups else {}
    for name, weights in weightings.items():  # option B
        weighted_mean = float(np.average(y_fit, weights=weights))
        predictions[f"B_weight_{name}"] = group_mean_predict(
            fine_fit, y_fit, fine_val, weights=weights, fallback=weighted_mean
        )
    parent_pred = group_mean_predict(parent_fit, y_fit, parent_val)
    for strength in SHRINK_STRENGTHS if "C" in groups else []:  # option C: shrink
        predictions[f"C_shrink_k{strength}"] = shrunk_group_mean_predict(
            fine_fit, y_fit, fine_val, parent_pred, strength
        )
    sqrt_weights = np.sqrt(total_fit)  # option C with noise aware evidence per row
    parent_weighted = group_mean_predict(
        parent_fit, y_fit, parent_val, weights=sqrt_weights,
        fallback=float(np.average(y_fit, weights=sqrt_weights)),
    )
    for strength in WEIGHTED_STRENGTHS if "Cw" in groups else []:
        predictions[f"C_shrink_sqrtw_k{strength}"] = shrunk_group_mean_predict(
            fine_fit, y_fit, fine_val, parent_weighted, strength, weights=sqrt_weights
        )
    scores = {
        name: score_predictions(total_val, y_val, pred) for name, pred in predictions.items()
    }
    print(f"parent cell mean (1e-5) alone: {score_predictions(total_val, y_val, parent_pred)}")
    print(to_markdown(pd.DataFrame(scores).T))


if __name__ == "__main__":
    main()
