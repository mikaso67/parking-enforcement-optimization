import numpy as np
import pandas as pd
import pytest

from parking.metrics import spearman, spearman_by_bucket, top_k_precision


def test_spearman_perfect() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert spearman(y_true, y_pred) == pytest.approx(1.0)


def test_spearman_reversed() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert spearman(y_true, y_pred) == pytest.approx(-1.0)


def test_spearman_constant() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [5.0, 5.0, 5.0, 5.0]
    assert np.isnan(spearman(y_true, y_pred))


def test_spearman_empty_and_insufficient() -> None:
    assert np.isnan(spearman([], []))
    assert np.isnan(spearman([1.0], [2.0]))


def test_spearman_by_bucket() -> None:
    df = pd.DataFrame(
        {
            "total_count": [1, 1, 3, 4, 10, 10, 10],
            "y_true": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0],
            "y_pred": [10.0, 20.0, 20.0, 10.0, 5.0, 5.0, 5.0],  # bucket 6..44 has constant y_pred
        }
    )
    result = spearman_by_bucket(df)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["bucket", "n_rows", "spearman"]
    assert list(result["bucket"]) == ["1", "2 to 5", "6 to 44", "45 and above"]

    # Bucket '1': 2 rows, perfect correlation
    row_1 = result[result["bucket"] == "1"].iloc[0]
    assert row_1["n_rows"] == 2
    assert row_1["spearman"] == pytest.approx(1.0)

    # Bucket '2 to 5': 2 rows, reversed correlation
    row_2_5 = result[result["bucket"] == "2 to 5"].iloc[0]
    assert row_2_5["n_rows"] == 2
    assert row_2_5["spearman"] == pytest.approx(-1.0)

    # Bucket '6 to 44': 3 rows, constant y_pred -> NaN
    row_6_44 = result[result["bucket"] == "6 to 44"].iloc[0]
    assert row_6_44["n_rows"] == 3
    assert np.isnan(row_6_44["spearman"])

    # Bucket '45 and above': 0 rows -> NaN
    row_45_plus = result[result["bucket"] == "45 and above"].iloc[0]
    assert row_45_plus["n_rows"] == 0
    assert np.isnan(row_45_plus["spearman"])


def test_top_k_precision_perfect() -> None:
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    assert top_k_precision(y_true, y_pred, k=2) == pytest.approx(1.0)


def test_top_k_precision_reversed() -> None:
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [4.0, 3.0, 2.0, 1.0]
    assert top_k_precision(y_true, y_pred, k=2) == pytest.approx(0.0)


def test_top_k_precision_constant() -> None:
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [5.0, 5.0, 5.0, 5.0]
    val = top_k_precision(y_true, y_pred, k=2)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_top_k_precision_k_larger_than_data() -> None:
    y_true = [10.0, 20.0, 30.0]
    y_pred = [1.0, 2.0, 3.0]
    assert top_k_precision(y_true, y_pred, k=10) == pytest.approx(1.0)


def test_top_k_precision_empty() -> None:
    assert top_k_precision([], [], k=5) == 0.0


def test_top_k_precision_ties_do_not_depend_on_row_order() -> None:
    y_true = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    swapped = np.array([1, 0, 2, 3, 4])
    assert top_k_precision(y_true, y_pred, k=1) == pytest.approx(1.0)
    assert top_k_precision(y_true[swapped], y_pred[swapped], k=1) == pytest.approx(1.0)


def test_top_k_precision_all_tied_maxima_count_as_true() -> None:
    y_true = [1.0] * 6 + [0.2] * 4
    y_pred = [10.0 - i for i in range(10)]
    assert top_k_precision(y_true, y_pred, k=3) == pytest.approx(1.0)


def test_top_k_precision_permutation_invariance() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=200).astype(float)
    y_pred = rng.integers(0, 3, size=200).astype(float)
    order = rng.permutation(200)
    base = top_k_precision(y_true, y_pred, k=20)
    assert top_k_precision(y_true[order], y_pred[order], k=20) == pytest.approx(base)
    assert 0.0 <= base <= 1.0


def test_top_k_precision_constant_prediction_is_chance_level() -> None:
    y_true = np.array([1.0] * 4 + [0.0] * 6)
    y_pred = np.zeros(10)
    # true set has 4 rows out of 10, so a constant prediction scores 4 / 10
    assert top_k_precision(y_true, y_pred, k=3) == pytest.approx(0.4)


def test_top_k_precision_perfect_and_reversed_without_ties() -> None:
    y = np.arange(10.0)
    assert top_k_precision(y, y, k=3) == pytest.approx(1.0)
    assert top_k_precision(y, -y, k=3) == pytest.approx(0.0)
