import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman(y_true: Any, y_pred: Any) -> float:
    """Compute Spearman rank correlation coefficient using scipy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) < 2 or len(y_p) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spearmanr(y_t, y_p)
    return float(res.statistic)


def spearman_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation for each total_count bucket.

    Buckets are defined as:
      - '1': total_count == 1
      - '2 to 5': 2 <= total_count <= 5
      - '6 to 44': 6 <= total_count <= 44
      - '45 and above': total_count >= 45

    Returns a DataFrame with columns ['bucket', 'n_rows', 'spearman'].
    Empty or constant buckets yield NaN for spearman without raising errors.
    """
    bucket_definitions = [
        ("1", df["total_count"] == 1),
        ("2 to 5", (df["total_count"] >= 2) & (df["total_count"] <= 5)),
        ("6 to 44", (df["total_count"] >= 6) & (df["total_count"] <= 44)),
        ("45 and above", df["total_count"] >= 45),
    ]
    rows = []
    for label, mask in bucket_definitions:
        sub_df = df[mask]
        n_rows = len(sub_df)
        sp = spearman(sub_df["y_true"], sub_df["y_pred"]) if n_rows > 0 else float("nan")
        rows.append({"bucket": label, "n_rows": n_rows, "spearman": sp})
    return pd.DataFrame(rows)


def _top_k_credit(values: np.ndarray, k: int) -> np.ndarray:
    """Share of the top k each row holds, with fractional credit for ties at the cutoff."""
    threshold = np.sort(values)[-k]
    above = values > threshold
    tied = values == threshold
    credit = above.astype(float)
    credit[tied] = (k - above.sum()) / tied.sum()
    return credit


def top_k_precision(y_true: Any, y_pred: Any, k: int) -> float:
    """Share of the predicted top k that belongs to the true top k, independent of row order.

    The true set holds every row whose y_true is at least the k-th largest value, so all
    ties at the cutoff count as true. Ties in y_pred at the cutoff share the remaining
    slots equally (fractional credit).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    n = len(y_t)
    if n == 0 or k <= 0:
        return 0.0
    k_eff = min(k, n)
    in_true_top = y_t >= np.sort(y_t)[-k_eff]
    credit = _top_k_credit(y_p, k_eff)
    return float(credit[in_true_top].sum() / k_eff)
