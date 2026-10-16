import pandas as pd
import pytest

from parking.data import FEATURE_DTYPES, TARGET, load_train


def write_sample(tmp_path, n=5, shift_index=False):
    X = pd.DataFrame({col: range(1, n + 1) for col in FEATURE_DTYPES})
    y = pd.DataFrame({TARGET: [0.5] * n})
    if shift_index:
        y.index = y.index + 1
    X.to_csv(tmp_path / "x_train.csv")
    y.to_csv(tmp_path / "y_train.csv")


def test_load_train_returns_aligned_frames(tmp_path):
    write_sample(tmp_path)
    X, y = load_train(tmp_path)
    assert list(X.columns) == list(FEATURE_DTYPES)
    assert len(X) == len(y) == 5
    assert X["hour"].dtype == "int8"


def test_load_train_rejects_misaligned_index(tmp_path):
    write_sample(tmp_path, shift_index=True)
    with pytest.raises(ValueError):
        load_train(tmp_path)
