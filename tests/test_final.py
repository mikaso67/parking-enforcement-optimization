import numpy as np
import pandas as pd

from parking.final import stack_train_test, to_submission


def small_frames():
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0.1, 0.2, 0.3], name="invalid_ratio", dtype="float32")
    X_test = pd.DataFrame({"a": [4, 5]})
    return X_train, y_train, X_test


def test_stack_keeps_train_first_and_unique_indexes():
    X_all, y_all, fit, val = stack_train_test(*small_frames())
    assert X_all.index.is_unique
    assert list(fit) == [True, True, True, False, False]
    assert list(val) == [False, False, False, True, True]
    assert list(X_all["a"]) == [1, 2, 3, 4, 5]


def test_test_targets_are_missing_and_train_targets_untouched():
    _, y_all, fit, val = stack_train_test(*small_frames())
    assert y_all[val].isna().all()
    assert np.allclose(y_all[fit], [0.1, 0.2, 0.3])


def test_submission_uses_the_test_index_and_target_name():
    out = to_submission(pd.Series([0.4, 0.6], index=[3, 4]), pd.Index([0, 1]))
    assert list(out.index) == [0, 1]
    assert list(out.columns) == ["invalid_ratio"]
    assert list(out["invalid_ratio"]) == [0.4, 0.6]
