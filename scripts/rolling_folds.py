"""Experiment 6: rolling origin folds for the statistical reference and LightGBM, seed 42.

Run from the repository root, one fold and one scheme per call on a small machine:
    PYTHONPATH=src python scripts/rolling_folds.py <fold> <scheme> [n_trees] [hour]
With the word hour as last argument, the cell by hour statistic is added to the features.
Folds (fit months, validation months): 1 is (1-3, 4-6), 2 is (1-6, 7-9), 3 is (1-9, 10-12).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from train_lgbm import PARAMS

from parking.baselines import score_predictions, to_markdown
from parking.data import load_train
from parking.oof_features import build_frames
from parking.tuning import fold_masks


def main() -> None:
    fold, scheme = int(sys.argv[1]), sys.argv[2]
    options = sys.argv[3:]
    hour_stat = "hour" in options
    numbers = [o for o in options if o != "hour"]
    n_trees = int(numbers[0]) if numbers else 100
    X, y = load_train(Path("data/raw"))
    fit, val = fold_masks(X["month_of_year"], fold)
    fit_frame, val_frame = build_frames(X, y, fit, val, scheme, hour_stat=hour_stat)
    features = [c for c in fit_frame.columns if c not in ("y", "month_of_year")]
    model = lgb.LGBMRegressor(n_estimators=n_trees, **PARAMS)
    model.fit(fit_frame[features], fit_frame["y"])
    name = f"fold{fold}_{scheme}" + ("_hour" if hour_stat else "")
    scores = {
        f"lgbm_{name}": score_predictions(
            val_frame["total_count"], val_frame["y"], model.predict(val_frame[features])
        ),
        f"mean_l4_{name}": score_predictions(
            val_frame["total_count"], val_frame["y"], val_frame["mean_l4"].to_numpy()
        ),
    }
    print(f"fit rows: {len(fit_frame)}, validation rows: {len(val_frame)}")
    print(to_markdown(pd.DataFrame(scores).T))


if __name__ == "__main__":
    main()
