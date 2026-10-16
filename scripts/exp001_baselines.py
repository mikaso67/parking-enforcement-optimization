"""Experiment 1: baseline predictors on the time split of decision D5 (seed 42).

Run from the repository root: PYTHONPATH=src python scripts/exp001_baselines.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from parking.baselines import group_mean_predict, score_predictions, to_markdown
from parking.data import load_train
from parking.zones import assign_cells

CELL_SIZES = {"1e-4": 1e-4, "3e-5": 3e-5, "1e-5": 1e-5}


def main() -> None:
    X, y = load_train(Path("data/raw"))
    fit, val = X["month_of_year"] <= 9, X["month_of_year"] >= 10
    X_fit, y_fit, X_val, y_val = X[fit], y[fit], X[val], y[val]
    predictions = {
        "global_mean": np.full(len(X_val), y_fit.mean()),
        "hour_mean": group_mean_predict(X_fit["hour"], y_fit, X_val["hour"]),
        "neg_total_count": -X_val["total_count"].to_numpy(dtype=float),
    }
    for label, size in CELL_SIZES.items():
        k_fit, k_val = assign_cells(X_fit, size), assign_cells(X_val, size)
        predictions[f"cell_mean_{label}"] = group_mean_predict(k_fit, y_fit, k_val)
        predictions[f"cell_mean_weighted_{label}"] = group_mean_predict(
            k_fit, y_fit, k_val, weights=X_fit["total_count"]
        )
    rows = {
        name: score_predictions(X_val["total_count"], y_val, pred)
        for name, pred in predictions.items()
    }
    print(f"validation rows: {len(X_val)}")
    print(X_val["total_count"].pipe(pd.cut, [0, 1, 5, 44, np.inf]).value_counts().sort_index())
    print(to_markdown(pd.DataFrame(rows).T))


if __name__ == "__main__":
    main()
