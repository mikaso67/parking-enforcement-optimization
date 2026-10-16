"""Final model: fit on all 12 training months, predict the test set, write the submission file.

Run from the repository root:
    PYTHONPATH=src python scripts/final_predict.py [n_trees]
Writes data/processed/y_test_pred.csv (index and invalid_ratio, same layout as y_train.csv)
and data/processed/test_features.parquet (features and prediction, for the map).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from train_lgbm import PARAMS

from parking.data import load_test, load_train
from parking.final import stack_train_test, to_submission
from parking.oof_features import build_frames

SCHEME = "expanding"


def main() -> None:
    n_trees = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    raw, out = Path("data/raw"), Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_train(raw)
    X_test = load_test(raw)
    X_all, y_all, fit, val = stack_train_test(X_train, y_train, X_test)
    del X_train, y_train
    fit_frame, test_frame = build_frames(X_all, y_all, fit, val, SCHEME)
    features = [c for c in fit_frame.columns if c not in ("y", "month_of_year")]
    model = lgb.LGBMRegressor(n_estimators=n_trees, **PARAMS)
    model.fit(fit_frame[features], fit_frame["y"])
    test_frame = test_frame.set_axis(test_frame.index - len(fit_frame))
    test_frame["prediction"] = model.predict(test_frame[features])
    to_submission(test_frame["prediction"], X_test.index).to_csv(out / "y_test_pred.csv")
    keep = ["mean_l4", "count_l4", "prediction", "total_count"]
    keep += ["hour", "day_of_week", "month_of_year"]
    extra = X_test[["longitude_scaled", "latitude_scaled"]]
    pd.concat([test_frame[keep], extra], axis=1).to_parquet(out / "test_features.parquet")
    print(f"fit rows: {len(fit_frame)}, test rows: {len(test_frame)}")
    print(test_frame["prediction"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
