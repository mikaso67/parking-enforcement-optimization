import numpy as np
import pandas as pd
import pytest

from parking.baselines import multilevel_shrunk_predict
from parking.target_stats import (
    NO_HISTORY_PRIOR,
    fit_side_stats,
    new_side_stats,
    to_codes,
)


def synthetic(n_fit: int = 60, n_new: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = n_fit + n_new
    coarse = pd.Series(rng.integers(0, 3, n))
    fine = pd.Series(coarse.to_numpy() * 10 + rng.integers(0, 4, n))
    y = rng.random(n_fit)
    month = np.repeat([1, 2, 3], n_fit // 3)
    return coarse, fine, y, month, n_fit


def test_new_side_stats_matches_multilevel_shrunk_predict() -> None:
    coarse, fine, y, _, n_fit = synthetic()
    codes = to_codes([coarse, fine], n_fit)
    predictions, _ = new_side_stats(codes, y, n_fit, strength=3.0)
    expected = multilevel_shrunk_predict(
        [coarse.iloc[:n_fit], fine.iloc[:n_fit]],
        pd.Series(y),
        [coarse.iloc[n_fit:].reset_index(drop=True), fine.iloc[n_fit:].reset_index(drop=True)],
        strength=3.0,
    )
    assert np.allclose(predictions[:, -1], expected)


def test_new_side_stats_counts_are_fit_row_counts_per_cell() -> None:
    coarse, fine, y, _, n_fit = synthetic()
    codes = to_codes([fine], n_fit)
    _, counts = new_side_stats(codes, y, n_fit, strength=1.0)
    expected = fine.iloc[n_fit:].map(fine.iloc[:n_fit].value_counts()).fillna(0).to_numpy()
    assert np.allclose(counts[:, 0], expected)


@pytest.mark.parametrize("scheme", ["leave_one_month_out", "expanding"])
def test_fit_side_stats_do_not_depend_on_the_rows_own_target(scheme: str) -> None:
    coarse, fine, y, month, n_fit = synthetic()
    codes = to_codes([coarse, fine], n_fit)
    base, _ = fit_side_stats(codes, y, month, n_fit, 3.0, scheme)
    changed = y.copy()
    changed[month == 2] = 1.0 - changed[month == 2]
    after, _ = fit_side_stats(codes, changed, month, n_fit, 3.0, scheme)
    assert np.allclose(base[month == 2], after[month == 2])
    assert not np.allclose(base[month != 2], after[month != 2])


def test_leave_one_month_out_hand_example() -> None:
    codes = [np.array([0, 0, 0, 0])]
    y = np.array([1.0, 0.0, 1.0, 1.0])
    month = np.array([1, 1, 2, 3])
    predictions, counts = fit_side_stats(codes, y, month, 4, 0.0)
    # month 1 rows use months 2 and 3: mean 1.0 on 2 rows
    assert np.allclose(predictions[:2, 0], 1.0)
    assert np.allclose(counts[:2, 0], 2.0)
    # month 3 row uses months 1 and 2: (1 + 0 + 1) / 3
    assert predictions[3, 0] == pytest.approx(2.0 / 3.0)


def test_expanding_scheme_uses_only_earlier_months_and_a_constant_for_the_first() -> None:
    codes = [np.array([0, 0, 0, 0])]
    y = np.array([1.0, 0.0, 1.0, 0.0])
    month = np.array([1, 1, 2, 3])
    predictions, counts = fit_side_stats(codes, y, month, 4, 0.0, "expanding")
    assert np.allclose(counts[:, 0], [0, 0, 2, 3])
    assert predictions[0, 0] == pytest.approx(NO_HISTORY_PRIOR)
    assert predictions[2, 0] == pytest.approx(0.5)
    assert predictions[3, 0] == pytest.approx(2.0 / 3.0)


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="scheme"):
        fit_side_stats([np.array([0])], np.array([1.0]), np.array([1]), 1, 1.0, "random")


def test_to_codes_vocabulary_ignores_the_new_rows() -> None:
    keys = pd.Series([5, 7, 5, 9, 7, 11])
    with_new = to_codes([keys], n_fit=3)[0]
    fit_only = to_codes([keys.iloc[:3]], n_fit=3)[0]
    assert (with_new[:3] == fit_only).all()
    assert with_new[3] == with_new[5] == 2  # unseen keys 9 and 11 share the extra code
    assert with_new[4] == with_new[1]  # key 7 was seen in the fit rows


def test_leave_one_month_out_warns_and_expanding_does_not() -> None:
    import warnings

    coarse, fine, y, month, n_fit = synthetic()
    codes = to_codes([coarse, fine], n_fit)
    with pytest.warns(UserWarning, match="not a forward-time validation"):
        fit_side_stats(codes, y, month, n_fit, 10.0, "leave_one_month_out")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_side_stats(codes, y, month, n_fit, 10.0, "expanding")
