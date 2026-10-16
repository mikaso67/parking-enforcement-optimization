"""Build the out of fold cell statistics for the fit and validation parts (D5, D7, D8).

Run from the repository root:
    PYTHONPATH=src python scripts/build_features.py [leave_one_month_out|expanding]
Writes data/interim/features_<scheme>_{fit,val}.parquet (ignored by git).
"""
import sys
from pathlib import Path

from parking.data import load_train
from parking.oof_features import build_frames


def main() -> None:
    scheme = sys.argv[1] if len(sys.argv) > 1 else "leave_one_month_out"
    X, y = load_train(Path("data/raw"))
    fit, val = X["month_of_year"] <= 9, X["month_of_year"] >= 10
    out = Path("data/interim")
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("fit", "val"), build_frames(X, y, fit, val, scheme), strict=True):
        frame.to_parquet(out / f"features_{scheme}_{name}.parquet")
        print(name, frame.shape)


if __name__ == "__main__":
    main()
