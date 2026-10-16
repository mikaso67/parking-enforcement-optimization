"""Experiment 4: multi level shrinkage and a total_count effect, D5 split, seed 42.

Reference: experiment 3 C_shrink_k3 (cell 1e-6 shrunk toward the 1e-5 cell mean, strength 3).
Run from the repository root: PYTHONPATH=src python scripts/exp004_multilevel.py [group ...]
Groups: L2, L3, L4, count (default: all). Run them one by one on a small machine.
"""
import sys
from pathlib import Path

import pandas as pd

from parking.baselines import (
    count_effect_predict,
    multilevel_shrunk_predict,
    score_predictions,
    to_markdown,
)
from parking.data import load_train
from parking.zones import assign_cells, nest_keys

SIZES = {"L2": [1e-5, 1e-6], "L3": [1e-5, 3e-6, 1e-6], "L4": [1e-5, 3e-6, 1e-6, 5e-7]}
COUNT_STRENGTH = 10
STRENGTHS = {"L2": [3], "L3": [3], "L4": [1, 3, 10, 30]}


def main() -> None:
    X, y = load_train(Path("data/raw"))
    fit, val = X["month_of_year"] <= 9, X["month_of_year"] >= 10
    X_fit, y_fit, X_val, y_val = X[fit], y[fit], X[val], y[val]
    total_fit, total_val = X_fit["total_count"], X_val["total_count"]
    groups = set(sys.argv[1:]) or {"L2", "L3", "L4", "count"}
    cache: dict[tuple[float, ...], tuple[pd.Series, pd.Series]] = {}

    def keys(sizes: tuple[float, ...]) -> tuple[pd.Series, pd.Series]:
        """Keys of the last size nested in the coarser ones, built row by row for fit and val."""
        if sizes not in cache:
            built = []
            for rows in (X_fit, X_val):
                current = assign_cells(rows, sizes[0])
                for size in sizes[1:]:
                    current = nest_keys(current, assign_cells(rows, size))
                built.append(current)
            cache[sizes] = (built[0], built[1])
        return cache[sizes]

    def predict(name: str, strength: float):
        pairs = [keys(tuple(SIZES[name][: i + 1])) for i in range(len(SIZES[name]))]
        levels_fit = [fit_keys for fit_keys, _ in pairs]
        levels_new = [new_keys for _, new_keys in pairs]
        return multilevel_shrunk_predict(levels_fit, y_fit, levels_new, strength)

    predictions = {}
    for name in ("L2", "L3", "L4"):
        if name in groups:
            for strength in STRENGTHS[name]:
                predictions[f"{name}_k{strength}"] = predict(name, strength)
    if "count" in groups:
        base = predict("L4", COUNT_STRENGTH)
        effect = count_effect_predict(total_fit, y_fit, total_val)
        for weight in (0.5, 1.0):
            predictions[f"L4_k{COUNT_STRENGTH}_plus_count_x{weight}"] = base + weight * effect
    scores = {
        label: score_predictions(total_val, y_val, pred) for label, pred in predictions.items()
    }
    print(to_markdown(pd.DataFrame(scores).T))


if __name__ == "__main__":
    main()
