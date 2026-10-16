import numpy as np
import pandas as pd

from parking.metrics import spearman, spearman_by_bucket


def group_mean_predict(
    keys_fit: pd.Series,
    y_fit: pd.Series,
    keys_new: pd.Series,
    weights: pd.Series | None = None,
    fallback: float | None = None,
) -> np.ndarray:
    """Predict the (optionally weighted) mean target of each key seen in the fit part.

    A key absent from the fit part gets `fallback`, by default the plain mean of the fit target.
    """
    if fallback is None:
        fallback = float(y_fit.mean())
    fit = pd.DataFrame({"key": keys_fit.to_numpy(), "y": y_fit.to_numpy(dtype=float)})
    if weights is None:
        means = fit.groupby("key")["y"].mean()
    else:
        fit["w"] = weights.to_numpy(dtype=float)
        fit["wy"] = fit["w"] * fit["y"]
        grouped = fit.groupby("key")[["wy", "w"]].sum()
        means = grouped["wy"] / grouped["w"]
    return keys_new.map(means).fillna(fallback).to_numpy(dtype=float)


def shrunk_group_mean_predict(
    keys_fit: pd.Series,
    y_fit: pd.Series,
    keys_new: pd.Series,
    prior_new: np.ndarray,
    strength: float,
    weights: pd.Series | None = None,
) -> np.ndarray:
    """Group mean pulled toward a prior.

    Prediction = (sum of weighted y + strength * prior) / (sum of weights + strength).

    The prior is given per new row (for example the mean of a coarser cell). A key absent from
    the fit part gets exactly its prior. Weights default to 1 per row.
    """
    w = np.ones(len(y_fit)) if weights is None else weights.to_numpy(dtype=float)
    fit = pd.DataFrame({"key": keys_fit.to_numpy(), "wy": w * y_fit.to_numpy(dtype=float), "w": w})
    grouped = fit.groupby("key")[["wy", "w"]].sum()
    sum_wy = keys_new.map(grouped["wy"]).fillna(0.0).to_numpy(dtype=float)
    sum_w = keys_new.map(grouped["w"]).fillna(0.0).to_numpy(dtype=float)
    denominator = sum_w + strength
    safe = np.where(denominator == 0, 1.0, denominator)
    return np.where(denominator == 0, prior_new, (sum_wy + strength * prior_new) / safe)


def multilevel_shrunk_predict(
    levels_fit: list[pd.Series],
    y_fit: pd.Series,
    levels_new: list[pd.Series],
    strength: float,
) -> np.ndarray:
    """Shrink level by level, from the coarsest keys to the finest.

    The prior of the first level is the fit mean. The prior of each next level is the
    prediction of the previous one, so a finer cell with no fit rows falls back to its parent.
    """
    prediction = np.full(len(levels_new[0]), float(y_fit.mean()))
    for keys_fit, keys_new in zip(levels_fit, levels_new, strict=True):
        prediction = shrunk_group_mean_predict(keys_fit, y_fit, keys_new, prediction, strength)
    return prediction


def count_effect_predict(
    total_fit: pd.Series, y_fit: pd.Series, total_new: pd.Series, cap: int = 60
) -> np.ndarray:
    """Mean target by capped total_count minus the overall fit mean, looked up for new rows."""
    y = y_fit.to_numpy(dtype=float)
    effect = pd.Series(y - y.mean()).groupby(total_fit.clip(upper=cap).to_numpy()).mean()
    return total_new.clip(upper=cap).map(effect).fillna(0.0).to_numpy(dtype=float)


def score_predictions(
    total_count: pd.Series, y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Spearman overall and by total_count bucket. A constant prediction scores 0.0."""
    frame = pd.DataFrame({"total_count": total_count, "y_true": y_true, "y_pred": y_pred})
    overall = 0.0 if np.ptp(y_pred) == 0 else spearman(y_true, y_pred)
    buckets = spearman_by_bucket(frame)["spearman"].fillna(0.0).tolist()
    return dict(zip(["overall", "1", "2-5", "6-44", "45+"], [overall, *buckets], strict=True))


def to_markdown(table: pd.DataFrame) -> str:
    """Render a table indexed by predictor name as a Markdown table with 3 decimals."""
    header = "| predictor | " + " | ".join(table.columns) + " |"
    divider = "|" + "---|" * (len(table.columns) + 1)
    lines = [
        f"| {name} | " + " | ".join(f"{value:.3f}" for value in row) + " |"
        for name, row in table.iterrows()
    ]
    return "\n".join([header, divider, *lines])
