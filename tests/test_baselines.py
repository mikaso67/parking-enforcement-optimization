import numpy as np
import pandas as pd
import pytest

from parking.baselines import (
    count_effect_predict,
    group_mean_predict,
    multilevel_shrunk_predict,
    score_predictions,
    shrunk_group_mean_predict,
    to_markdown,
)


def test_group_mean_predict_uses_group_means() -> None:
    keys = pd.Series(["a", "a", "b"])
    y = pd.Series([0.0, 1.0, 0.5])
    pred = group_mean_predict(keys, y, pd.Series(["b", "a"]))
    assert np.allclose(pred, [0.5, 0.5])


def test_group_mean_predict_unseen_key_gets_fit_mean() -> None:
    keys = pd.Series(["a", "b"])
    y = pd.Series([0.2, 0.6])
    pred = group_mean_predict(keys, y, pd.Series(["zzz"]))
    assert pred[0] == pytest.approx(0.4)


def test_group_mean_predict_weighted_mean() -> None:
    keys = pd.Series(["a", "a"])
    y = pd.Series([0.0, 1.0])
    weights = pd.Series([1, 3])
    pred = group_mean_predict(keys, y, pd.Series(["a"]), weights=weights)
    assert pred[0] == pytest.approx(0.75)


def test_group_mean_predict_ignores_new_rows_and_index() -> None:
    keys = pd.Series([1, 1, 2], index=[10, 11, 12])
    y = pd.Series([0.0, 0.4, 1.0], index=[10, 11, 12])
    new = pd.Series([2, 1], index=[500, 600])
    assert np.allclose(group_mean_predict(keys, y, new), [1.0, 0.2])


def test_score_predictions_perfect_ranking_scores_one_everywhere() -> None:
    total = pd.Series([1, 1, 3, 4, 10, 12, 50, 60])
    y = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    scores = score_predictions(total, y, y.to_numpy())
    assert set(scores) == {"overall", "1", "2-5", "6-44", "45+"}
    assert all(value == pytest.approx(1.0) for value in scores.values())


def test_score_predictions_constant_prediction_scores_zero() -> None:
    total = pd.Series([1, 2, 10, 50])
    y = pd.Series([0.1, 0.5, 0.3, 0.9])
    scores = score_predictions(total, y, np.full(4, 0.5))
    assert all(value == 0.0 for value in scores.values())


def test_score_predictions_empty_bucket_scores_zero() -> None:
    total = pd.Series([1, 1, 1])
    y = pd.Series([0.0, 0.5, 1.0])
    scores = score_predictions(total, y, y.to_numpy())
    assert scores["overall"] == pytest.approx(1.0)
    assert scores["45+"] == 0.0


def test_to_markdown_has_header_divider_and_three_decimals() -> None:
    table = pd.DataFrame({"overall": [0.5], "1": [0.12345]}, index=["a"])
    lines = to_markdown(table).split("\n")
    assert lines[0] == "| predictor | overall | 1 |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| a | 0.500 | 0.123 |"


def test_group_mean_predict_custom_fallback() -> None:
    pred = group_mean_predict(pd.Series(["a"]), pd.Series([0.2]), pd.Series(["zzz"]), fallback=0.9)
    assert pred[0] == pytest.approx(0.9)


def test_shrunk_mean_with_zero_strength_is_the_plain_mean() -> None:
    keys = pd.Series(["a", "a", "b"])
    y = pd.Series([0.0, 1.0, 0.5])
    new = pd.Series(["a", "b"])
    pred = shrunk_group_mean_predict(keys, y, new, prior_new=np.array([0.9, 0.9]), strength=0.0)
    assert np.allclose(pred, [0.5, 0.5])


def test_shrunk_mean_blends_group_mean_and_prior() -> None:
    keys = pd.Series(["a", "a"])
    y = pd.Series([1.0, 1.0])
    pred = shrunk_group_mean_predict(
        keys, y, pd.Series(["a"]), prior_new=np.array([0.0]), strength=2.0
    )
    # (2 * 1.0 + 2 * 0.0) / (2 + 2)
    assert pred[0] == pytest.approx(0.5)


def test_shrunk_mean_unknown_key_gets_exactly_the_prior() -> None:
    pred = shrunk_group_mean_predict(
        pd.Series(["a"]),
        pd.Series([1.0]),
        pd.Series(["zzz"]),
        prior_new=np.array([0.3]),
        strength=5.0,
    )
    assert pred[0] == pytest.approx(0.3)


def test_shrunk_mean_unknown_key_with_zero_strength_does_not_divide_by_zero() -> None:
    pred = shrunk_group_mean_predict(
        pd.Series(["a"]),
        pd.Series([1.0]),
        pd.Series(["zzz"]),
        prior_new=np.array([0.3]),
        strength=0.0,
    )
    assert pred[0] == pytest.approx(0.3)


def test_shrunk_mean_with_weights_and_large_strength_approaches_the_prior() -> None:
    keys = pd.Series(["a", "a"])
    y = pd.Series([0.0, 1.0])
    weights = pd.Series([1, 3])
    near_prior = shrunk_group_mean_predict(
        keys, y, pd.Series(["a"]), np.array([0.2]), strength=1e9, weights=weights
    )
    weighted = shrunk_group_mean_predict(
        keys, y, pd.Series(["a"]), np.array([0.2]), strength=0.0, weights=weights
    )
    assert near_prior[0] == pytest.approx(0.2, abs=1e-6)
    assert weighted[0] == pytest.approx(0.75)


def test_multilevel_single_level_matches_shrinkage_toward_the_fit_mean() -> None:
    keys = pd.Series(["a", "a", "b"])
    y = pd.Series([1.0, 1.0, 0.0])
    new = pd.Series(["a", "b", "zzz"])
    pred = multilevel_shrunk_predict([keys], y, [new], strength=1.0)
    prior = y.mean()
    assert pred[0] == pytest.approx((2.0 + prior) / 3.0)
    assert pred[1] == pytest.approx((0.0 + prior) / 2.0)
    assert pred[2] == pytest.approx(prior)


def test_multilevel_unknown_fine_cell_falls_back_to_its_parent() -> None:
    coarse_fit = pd.Series(["p", "p"])
    fine_fit = pd.Series(["c1", "c1"])
    y = pd.Series([1.0, 0.0])
    coarse_new = pd.Series(["p"])
    fine_new = pd.Series(["unseen"])
    both = multilevel_shrunk_predict([coarse_fit, fine_fit], y, [coarse_new, fine_new], 2.0)
    parent_only = multilevel_shrunk_predict([coarse_fit], y, [coarse_new], 2.0)
    assert both[0] == pytest.approx(parent_only[0])


def test_multilevel_zero_strength_uses_the_finest_cell_mean() -> None:
    coarse = pd.Series(["p", "p", "p"])
    fine = pd.Series(["c1", "c1", "c2"])
    y = pd.Series([1.0, 1.0, 0.0])
    pred = multilevel_shrunk_predict([coarse, fine], y, [coarse, fine], 0.0)
    assert np.allclose(pred, [1.0, 1.0, 0.0])


def test_count_effect_is_centered_and_looks_up_by_count() -> None:
    total = pd.Series([1, 1, 5, 5])
    y = pd.Series([1.0, 0.8, 0.2, 0.0])
    effect = count_effect_predict(total, y, pd.Series([1, 5, 7]))
    assert effect[0] == pytest.approx(0.9 - 0.5)
    assert effect[1] == pytest.approx(0.1 - 0.5)
    assert effect[2] == 0.0


def test_count_effect_caps_large_counts() -> None:
    total = pd.Series([60, 100, 1])
    y = pd.Series([0.2, 0.4, 0.9])
    effect = count_effect_predict(total, y, pd.Series([60, 100, 500]), cap=60)
    assert effect[0] == pytest.approx(effect[1])
    assert effect[1] == pytest.approx(effect[2])
