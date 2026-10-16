from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

FEATURE_DTYPES = {
    "total_count": "int32",
    "longitude_scaled": "float32",
    "latitude_scaled": "float32",
    "Precipitations": "float32",
    "HauteurNeige": "float32",
    "Temperature": "float32",
    "ForceVent": "float32",
    "day_of_week": "int8",
    "month_of_year": "int8",
    "hour": "int8",
}
TARGET = "invalid_ratio"


def load_features(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, dtype=FEATURE_DTYPES)


def load_target(path: Path) -> pd.Series:
    return pd.read_csv(path, index_col=0, dtype={TARGET: "float32"})[TARGET]


def load_train(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    X = load_features(raw_dir / "x_train.csv")
    y = load_target(raw_dir / "y_train.csv")
    if not X.index.equals(y.index):
        raise ValueError("x_train and y_train indexes do not match")
    return X, y


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return load_features(raw_dir / "x_test.csv")
