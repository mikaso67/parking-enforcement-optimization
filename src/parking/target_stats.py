import warnings

import numpy as np
import pandas as pd

SCHEMES = ("leave_one_month_out", "expanding")
NO_HISTORY_PRIOR = 0.5  # constant, not learned from data, used when no earlier month exists


def _group_sums(codes: np.ndarray, y: np.ndarray, n_codes: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.bincount(codes, weights=y, minlength=n_codes),
        np.bincount(codes, minlength=n_codes).astype(float),
    )


def _shrink(sum_y: np.ndarray, count: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    denominator = count + strength
    safe = np.where(denominator == 0, 1.0, denominator)
    return np.where(denominator == 0, prior, (sum_y + strength * prior) / safe)


def to_codes(levels: list[pd.Series], n_fit: int) -> list[np.ndarray]:
    """Integer codes of each level's keys, with the vocabulary taken from the first n_fit rows.

    Rows after n_fit whose key never appears in the fit rows get one shared extra code, which
    has no fit rows behind it. Nothing outside the fit rows decides the codes.
    """
    coded = []
    for keys in levels:
        values = keys.to_numpy()
        fit_codes, vocabulary = pd.factorize(values[:n_fit])
        new_codes = pd.Index(vocabulary).get_indexer(values[n_fit:])
        new_codes[new_codes < 0] = len(vocabulary)
        coded.append(np.concatenate([fit_codes, new_codes]).astype("int64"))
    return coded


def fit_side_stats(
    codes: list[np.ndarray],
    y_fit: np.ndarray,
    month_fit: np.ndarray,
    n_fit: int,
    strength: float,
    scheme: str = "leave_one_month_out",
) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk cell means for the fit rows, computed without the row's own month.

    codes[l] holds the integer code of every row (fit rows first) at level l, coarse to fine.
    Returns (predictions, counts), both of shape (n_fit, n_levels). With the scheme
    leave_one_month_out a row of month m uses all fit months except m. With expanding it uses
    only the months before m. leave_one_month_out is a comparison scheme and not a validation,
    since a row then uses later months: reported scores use expanding.
    The prior of the first level is the mean of the rows used, or the constant NO_HISTORY_PRIOR
    when there are none (first month of the expanding scheme).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme}")
    if scheme == "leave_one_month_out":
        warnings.warn(
            "leave_one_month_out uses later months for a row's statistics: it is a comparison "
            "scheme, not a forward-time validation. Use expanding for any reported score.",
            UserWarning,
            stacklevel=2,
        )
    n_levels = len(codes)
    predictions = np.zeros((n_fit, n_levels))
    counts = np.zeros((n_fit, n_levels))
    for month in np.unique(month_fit):
        target = month_fit == month
        used = month_fit != month if scheme == "leave_one_month_out" else month_fit < month
        start = y_fit[used].mean() if used.any() else NO_HISTORY_PRIOR
        prior = np.full(int(target.sum()), start)
        for level, level_codes in enumerate(codes):
            fit_codes = level_codes[:n_fit]
            n_codes = int(level_codes.max()) + 1
            sum_y, count = _group_sums(fit_codes[used], y_fit[used], n_codes)
            rows = fit_codes[target]
            prior = _shrink(sum_y[rows], count[rows], prior, strength)
            predictions[target, level] = prior
            counts[target, level] = count[rows]
    return predictions, counts


def new_side_stats(
    codes: list[np.ndarray], y_fit: np.ndarray, n_fit: int, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk cell means for rows outside the fit part, computed from all fit rows."""
    n_new = len(codes[0]) - n_fit
    predictions = np.zeros((n_new, len(codes)))
    counts = np.zeros((n_new, len(codes)))
    prior = np.full(n_new, y_fit.mean())
    for level, level_codes in enumerate(codes):
        n_codes = int(level_codes.max()) + 1
        sum_y, count = _group_sums(level_codes[:n_fit], y_fit, n_codes)
        rows = level_codes[n_fit:]
        prior = _shrink(sum_y[rows], count[rows], prior, strength)
        predictions[:, level] = prior
        counts[:, level] = count[rows]
    return predictions, counts
