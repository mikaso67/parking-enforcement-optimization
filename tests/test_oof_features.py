import numpy as np
import pandas as pd
import pytest

from parking.oof_features import PASS_THROUGH, build_frames


def synthetic(n: int = 600, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "longitude_scaled": (rng.integers(0, 6, n) * 1e-5).astype("float32"),
            "latitude_scaled": (rng.integers(0, 6, n) * 1e-5).astype("float32"),
            "total_count": rng.integers(1, 50, n),
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "Precipitations": rng.random(n),
            "HauteurNeige": rng.random(n),
            "Temperature": rng.random(n),
            "ForceVent": rng.random(n),
            "month_of_year": rng.integers(1, 7, n),
        },
        index=pd.RangeIndex(100, 100 + n),
    )
    return X, pd.Series(rng.random(n), index=X.index)


def masks(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return X["month_of_year"] <= 4, X["month_of_year"] >= 5


@pytest.mark.parametrize("scheme", ["leave_one_month_out", "expanding"])
def test_frames_cover_the_right_rows_and_keep_the_target(scheme: str) -> None:
    X, y = synthetic()
    fit, val = masks(X)
    fit_frame, val_frame = build_frames(X, y, fit, val, scheme)
    assert fit_frame.index.equals(X.index[fit])
    assert val_frame.index.equals(X.index[val])
    assert np.allclose(fit_frame["y"], y[fit].to_numpy(), atol=1e-6)
    assert np.allclose(val_frame["y"], y[val].to_numpy(), atol=1e-6)
    assert set(PASS_THROUGH) <= set(fit_frame.columns)


@pytest.mark.parametrize("scheme", ["leave_one_month_out", "expanding"])
def test_validation_statistics_depend_on_fit_rows_only(scheme: str) -> None:
    X, y = synthetic()
    fit, val = masks(X)
    _, base = build_frames(X, y, fit, val, scheme)
    changed_y = y.copy()
    changed_y[val] = 1.0 - changed_y[val]
    _, after_val_change = build_frames(X, changed_y, fit, val, scheme)
    stat_columns = [c for c in base.columns if c.startswith(("mean_l", "count_l"))]
    assert np.allclose(base[stat_columns], after_val_change[stat_columns])
    changed_fit = y.copy()
    changed_fit[fit] = 1.0 - changed_fit[fit]
    _, after_fit_change = build_frames(X, changed_fit, fit, val, scheme)
    assert not np.allclose(base["mean_l4"], after_fit_change["mean_l4"])


def test_unseen_validation_cell_has_no_history() -> None:
    X, y = synthetic()
    fit, val = masks(X)
    X = X.copy()
    unseen = X.index[val][0]
    X.loc[unseen, ["longitude_scaled", "latitude_scaled"]] = np.float32(9e-3)
    _, val_frame = build_frames(X, y, fit, val, "expanding")
    assert val_frame.loc[unseen, "count_l4"] == 0


@pytest.mark.parametrize("scheme", ["leave_one_month_out", "expanding"])
def test_hour_statistic_adds_two_columns_and_nothing_else(scheme: str) -> None:
    X, y = synthetic()
    fit, val = masks(X)
    base_fit, _ = build_frames(X, y, fit, val, scheme)
    hour_fit, hour_val = build_frames(X, y, fit, val, scheme, hour_stat=True)
    assert set(hour_fit.columns) - set(base_fit.columns) == {"mean_l3h", "count_l3h"}
    assert np.allclose(hour_fit[base_fit.columns], base_fit)
    assert (hour_val["count_l3h"] <= hour_val["count_l3"]).all()


@pytest.mark.parametrize("scheme", ["leave_one_month_out", "expanding"])
def test_hour_statistic_depends_on_fit_rows_only(scheme: str) -> None:
    X, y = synthetic()
    fit, val = masks(X)
    _, base = build_frames(X, y, fit, val, scheme, hour_stat=True)
    changed = y.copy()
    changed[val] = 1.0 - changed[val]
    _, after = build_frames(X, changed, fit, val, scheme, hour_stat=True)
    assert np.allclose(base[["mean_l3h", "count_l3h"]], after[["mean_l3h", "count_l3h"]])


def test_hour_statistic_separates_hours_inside_a_cell() -> None:
    n = 800
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "longitude_scaled": np.zeros(n, dtype="float32"),
            "latitude_scaled": np.zeros(n, dtype="float32"),
            "total_count": rng.integers(1, 50, n),
            "hour": np.where((np.arange(n) // 4) % 2 == 0, 8, 17),
            "day_of_week": rng.integers(0, 7, n),
            "Precipitations": rng.random(n),
            "HauteurNeige": rng.random(n),
            "Temperature": rng.random(n),
            "ForceVent": rng.random(n),
            "month_of_year": np.tile([1, 2, 3, 4], n // 4),
        }
    )
    y = pd.Series(np.where(X["hour"] == 8, 0.2, 0.8))
    fit, val = X["month_of_year"] <= 3, X["month_of_year"] == 4
    _, val_frame = build_frames(X, y, fit, val, "expanding", hour_stat=True)
    morning = val_frame.loc[X["hour"][val] == 8, "mean_l3h"]
    evening = val_frame.loc[X["hour"][val] == 17, "mean_l3h"]
    assert morning.max() < 0.35 < 0.65 < evening.min()
    assert np.allclose(val_frame["mean_l3"], val_frame["mean_l3"].iloc[0])
