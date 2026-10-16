"""Experiment 2: finer cells against the 1e-5 reference (seed 42), D5 split.

Run from the repository root: PYTHONPATH=src python scripts/exp002_cell_size.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from parking.baselines import group_mean_predict, score_predictions, to_markdown
from parking.data import load_train
from parking.zones import assign_cells, point_keys

CELL_SIZES = {"1e-5": 1e-5, "5e-6": 5e-6, "3e-6": 3e-6, "1e-6": 1e-6}


def overall(total: pd.Series, y_true: pd.Series, pred: np.ndarray, mask: np.ndarray) -> float:
    return score_predictions(total[mask], y_true[mask], pred[mask])["overall"]


def main() -> None:
    X, y = load_train(Path("data/raw"))
    fit, val = X["month_of_year"] <= 9, X["month_of_year"] >= 10
    X_fit, y_fit, X_val, y_val = X[fit], y[fit], X[val], y[val]
    total = X_val["total_count"]
    keys = {
        label: (assign_cells(X_fit, size), assign_cells(X_val, size))
        for label, size in CELL_SIZES.items()
    }
    keys["exact_point"] = (point_keys(X_fit), point_keys(X_val))
    scores, coverage = {}, {}
    for label, (k_fit, k_val) in keys.items():
        pred = group_mean_predict(k_fit, y_fit, k_val)
        scores[f"cell_mean_{label}"] = score_predictions(total, y_val, pred)
        known = k_val.isin(set(k_fit.unique())).to_numpy()
        coverage[label] = {
            "fit_cells": k_fit.nunique(),
            "val_rows_in_known_cell": known.mean(),
            "spearman_known": overall(total, y_val, pred, known),
            "spearman_unknown": overall(total, y_val, pred, ~known),
        }
    print(to_markdown(pd.DataFrame(scores).T))
    print(pd.DataFrame(coverage).T.round(3).to_string())


if __name__ == "__main__":
    main()
