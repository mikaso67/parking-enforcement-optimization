"""Where the model ranks well or badly: Spearman by group, familiarity of the cell, calibration."""
import numpy as np
import pandas as pd

from parking.metrics import spearman

FAMILIARITY_BINS = [-0.5, 0.5, 4.5, 19.5, np.inf]
FAMILIARITY_LABELS = ["0", "1-4", "5-19", "20+"]
BUCKET_BINS = [0, 1, 5, 44, np.inf]
BUCKET_LABELS = ["1", "2-5", "6-44", "45+"]


def familiarity_group(history_count: pd.Series) -> pd.Series:
    """Group rows by the number of fit rows seen in their finest cell (0 means unseen)."""
    return pd.cut(history_count, FAMILIARITY_BINS, labels=FAMILIARITY_LABELS)


def bucket_group(total_count: pd.Series) -> pd.Series:
    """Group rows by the total_count buckets of the evaluation (1, 2-5, 6-44, 45+)."""
    return pd.cut(total_count, BUCKET_BINS, labels=BUCKET_LABELS)


def spearman_by_group(
    groups: pd.Series, y_true: pd.Series, y_pred: np.ndarray, min_rows: int = 1000
) -> pd.DataFrame:
    """Spearman, row count and means for each group. Groups below min_rows give NaN Spearman."""
    frame = pd.DataFrame(
        {"group": groups.reset_index(drop=True), "y": y_true.to_numpy(), "pred": y_pred}
    )
    rows = []
    for name, part in frame.groupby("group", observed=True):
        score = spearman(part["y"], part["pred"]) if len(part) >= min_rows else float("nan")
        rows.append((name, len(part), score, part["y"].mean(), part["pred"].mean()))
    return pd.DataFrame(rows, columns=["group", "n_rows", "spearman", "mean_y", "mean_pred"])


def calibration_by_decile(y_true: pd.Series, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean prediction and mean target in each decile of the prediction (ties broken by order)."""
    order = pd.Series(y_pred).rank(method="first")
    bins = pd.qcut(order, n_bins, labels=False)
    frame = pd.DataFrame({"bin": bins, "y": y_true.to_numpy(), "pred": y_pred})
    return frame.groupby("bin").agg(mean_pred=("pred", "mean"), mean_y=("y", "mean")).reset_index()


def extreme_share(y_true: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Share of rows whose target is exactly 0 and exactly 1, per group."""
    frame = pd.DataFrame({"group": groups.reset_index(drop=True), "y": y_true.to_numpy()})
    return frame.groupby("group", observed=True)["y"].agg(
        share_zero=lambda s: float((s == 0).mean()), share_one=lambda s: float((s == 1).mean())
    ).reset_index()


def history_counts(keys_fit: pd.Series, keys_new: pd.Series) -> np.ndarray:
    """Number of fit rows that share each new row's cell key (0 when the cell is unseen)."""
    counts = keys_fit.value_counts()
    return counts.reindex(keys_new.to_numpy()).fillna(0).to_numpy().astype("int64")


def reweighted_spearman(
    groups: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_share: dict[str, float],
    n_samples: int = 1_000_000,
    seed: int = 42,
) -> float:
    """Spearman on a resample of the rows whose group shares follow target_share.

    Answers: what would the score be if the groups were as frequent as in target_share?
    """
    rng = np.random.default_rng(seed)
    labels = groups.astype(str).to_numpy()
    positions = np.arange(len(labels))
    picked = []
    for name, share in target_share.items():
        pool = positions[labels == name]
        if len(pool) == 0 or share <= 0:
            continue
        picked.append(rng.choice(pool, size=int(round(share * n_samples)), replace=True))
    chosen = np.concatenate(picked)
    return spearman(y_true.to_numpy()[chosen], np.asarray(y_pred)[chosen])
