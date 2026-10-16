"""Final fit on all training months and prediction of the test set (frozen model, D10)."""
import pandas as pd


def stack_train_test(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stack train and test so that build_frames can treat them as fit and validation rows.

    Test rows get indexes after the last train index (both files start at 0) and a missing target.
    """
    offset = len(X_train)
    shifted = X_test.set_axis(X_test.index + offset)
    X_all = pd.concat([X_train, shifted])
    y_test = pd.Series(float("nan"), index=shifted.index, name=y_train.name, dtype="float32")
    y_all = pd.concat([y_train, y_test])
    fit = pd.Series(X_all.index < offset, index=X_all.index)
    return X_all, y_all, fit, ~fit


def to_submission(prediction: pd.Series, test_index: pd.Index) -> pd.DataFrame:
    """One column table named like the training target, indexed like the test file."""
    return pd.DataFrame({"invalid_ratio": prediction.to_numpy()}, index=test_index)
