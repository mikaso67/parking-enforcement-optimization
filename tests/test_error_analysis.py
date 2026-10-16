import numpy as np
import pandas as pd

from parking.error_analysis import (
    bucket_group,
    calibration_by_decile,
    extreme_share,
    familiarity_group,
    history_counts,
    reweighted_spearman,
    spearman_by_group,
)


def test_familiarity_group_edges() -> None:
    counts = pd.Series([0, 1, 4, 5, 19, 20, 500])
    expected = ["0", "1-4", "1-4", "5-19", "5-19", "20+", "20+"]
    assert list(familiarity_group(counts).astype(str)) == expected


def test_bucket_group_edges() -> None:
    totals = pd.Series([1, 2, 5, 6, 44, 45, 300])
    expected = ["1", "2-5", "2-5", "6-44", "6-44", "45+", "45+"]
    assert list(bucket_group(totals).astype(str)) == expected


def test_spearman_by_group_perfect_and_inverted() -> None:
    y = pd.Series(np.tile(np.arange(10.0), 2))
    pred = np.concatenate([np.arange(10.0), -np.arange(10.0)])
    groups = pd.Series(["a"] * 10 + ["b"] * 10)
    out = spearman_by_group(groups, y, pred, min_rows=5).set_index("group")
    assert np.isclose(out.loc["a", "spearman"], 1.0)
    assert np.isclose(out.loc["b", "spearman"], -1.0)
    assert out.loc["a", "n_rows"] == 10


def test_spearman_by_group_small_group_is_nan() -> None:
    y = pd.Series(np.arange(6.0))
    out = spearman_by_group(pd.Series(["a"] * 3 + ["b"] * 3), y, np.arange(6.0), min_rows=4)
    assert out["spearman"].isna().all()


def test_calibration_by_decile_is_monotone_for_a_good_model() -> None:
    rng = np.random.default_rng(42)
    pred = rng.random(1000)
    y = pd.Series(pred + rng.normal(0, 0.05, 1000))
    table = calibration_by_decile(y, pred)
    assert len(table) == 10
    assert table["mean_y"].is_monotonic_increasing
    assert (table["mean_pred"].diff().dropna() > 0).all()


def test_extreme_share() -> None:
    y = pd.Series([0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    out = extreme_share(y, groups).set_index("group")
    assert np.isclose(out.loc["a", "share_zero"], 2 / 3)
    assert out.loc["a", "share_one"] == 0.0
    assert out.loc["b", "share_one"] == 1.0


def test_history_counts_counts_fit_rows_per_cell() -> None:
    fit = pd.Series([5, 5, 5, 7, 9])
    new = pd.Series([5, 7, 8, 9, 5])
    assert list(history_counts(fit, new)) == [3, 1, 0, 1, 3]


def test_reweighted_spearman_follows_the_target_shares() -> None:
    rng = np.random.default_rng(42)
    n = 4000
    groups = pd.Series(["good"] * (n // 2) + ["bad"] * (n // 2))
    y = pd.Series(rng.random(n))
    pred = np.where(groups == "good", y + rng.normal(0, 0.05, n), rng.random(n))
    only_good = reweighted_spearman(groups, y, pred, {"good": 1.0}, n_samples=3000)
    only_bad = reweighted_spearman(groups, y, pred, {"bad": 1.0}, n_samples=3000)
    mixed = reweighted_spearman(groups, y, pred, {"good": 0.5, "bad": 0.5}, n_samples=3000)
    assert only_good > 0.9
    assert abs(only_bad) < 0.1
    assert only_bad < mixed < only_good


def test_groups_keep_the_declared_category_order() -> None:
    counts = pd.Series([25] * 3 + [10] * 3 + [0] * 3 + [2] * 3)
    y = pd.Series(np.arange(12.0))
    pred = np.arange(12.0)
    out = spearman_by_group(familiarity_group(counts), y, pred, min_rows=1)
    assert list(out["group"]) == ["0", "1-4", "5-19", "20+"]
    shares = extreme_share(y, familiarity_group(counts))
    assert list(shares["group"]) == ["0", "1-4", "5-19", "20+"]
    totals = pd.Series([1, 1, 3, 3, 10, 10, 60, 60, 1, 3, 10, 60])
    by_bucket = spearman_by_group(bucket_group(totals), y, pred, min_rows=1)
    assert list(by_bucket["group"]) == ["1", "2-5", "6-44", "45+"]
