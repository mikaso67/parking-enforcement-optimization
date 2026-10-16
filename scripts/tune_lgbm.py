"""Experiment 7: one at a time LightGBM tuning on the rolling folds, expanding scheme, seed 42.

Run from the repository root, one fold per call:
    PYTHONPATH=src python scripts/tune_lgbm.py <fold> [name=value ...]
Example: tune_lgbm.py 1 num_leaves=127 learning_rate=0.05
Trains once up to 400 trees (or max_trees=N) and scores the validation part at several tree counts.
The features are cached in data/interim so a sweep only pays for the training.
"""
import sys
from pathlib import Path

import lightgbm as lgb
from train_lgbm import PARAMS

from parking.baselines import score_predictions
from parking.data import load_train
from parking.tuning import cached_frames, parse_overrides

SCHEME = "expanding"
CHECKPOINTS = (50, 100, 200, 300, 400)
INTEGER_PARAMS = {"num_leaves", "min_child_samples", "max_trees"}


def main() -> None:
    fold = int(sys.argv[1])
    overrides = parse_overrides(sys.argv[2:], INTEGER_PARAMS)
    max_trees = overrides.pop("max_trees", max(CHECKPOINTS))
    fit, val = cached_frames(
        fold, SCHEME, Path("data/interim"), lambda: load_train(Path("data/raw"))
    )
    features = [c for c in fit.columns if c not in ("y", "month_of_year")]
    model = lgb.LGBMRegressor(n_estimators=max_trees, **{**PARAMS, **overrides})
    model.fit(fit[features], fit["y"])
    label = " ".join(sys.argv[2:]) or "reference"
    for trees in (t for t in CHECKPOINTS if t <= max_trees):
        pred = model.predict(val[features], num_iteration=trees)
        scores = score_predictions(val["total_count"], val["y"], pred)
        shown = " ".join(f"{k}={v:.3f}" for k, v in scores.items())
        print(f"fold{fold} | {label} | trees={trees} | {shown}")


if __name__ == "__main__":
    main()
