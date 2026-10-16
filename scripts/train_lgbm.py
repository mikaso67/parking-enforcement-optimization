"""Experiment 5: LightGBM on out of fold cell statistics (D7 option C), D5 split, seed 42.

Run from the repository root (after build_features.py):
    PYTHONPATH=src python scripts/train_lgbm.py [scheme] [n_train_rows] [n_trees]
The month of the row is not a feature (validation months never appear in the fit part).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from parking.baselines import score_predictions, to_markdown

SEED = 42
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 2,
    "verbose": -1,
}


def main() -> None:
    scheme = sys.argv[1] if len(sys.argv) > 1 else "leave_one_month_out"
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    n_trees = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    folder = Path("data/interim")
    fit = pd.read_parquet(folder / f"features_{scheme}_fit.parquet")
    val = pd.read_parquet(folder / f"features_{scheme}_val.parquet")
    if n_rows < len(fit):
        fit = fit.sample(n=n_rows, random_state=SEED)
    features = [c for c in fit.columns if c not in ("y", "month_of_year")]
    model = lgb.LGBMRegressor(n_estimators=n_trees, **PARAMS)
    model.fit(fit[features], fit["y"])
    pred = model.predict(val[features])
    scores = {f"lgbm_{scheme}_{len(fit)}rows_{n_trees}trees": score_predictions(
        val["total_count"], val["y"], pred
    )}
    stat_only = score_predictions(val["total_count"], val["y"], val["mean_l4"].to_numpy())
    scores["statistical_reference_l4"] = stat_only
    print(to_markdown(pd.DataFrame(scores).T))
    gain = pd.Series(model.booster_.feature_importance("gain"), index=features)
    print((gain / gain.sum()).sort_values(ascending=False).round(3).to_string())
    np.save(folder / "last_val_pred.npy", pred)


if __name__ == "__main__":
    main()
