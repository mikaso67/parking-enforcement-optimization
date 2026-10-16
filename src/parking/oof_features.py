"""Out of fold cell statistics and the feature frames built from them (D5, D7, D8)."""
import numpy as np
import pandas as pd

from parking.target_stats import fit_side_stats, new_side_stats, to_codes
from parking.zones import assign_cells, nest_keys

SIZES = [1e-5, 3e-6, 1e-6, 5e-7]
STRENGTH = 10.0
PASS_THROUGH = [
    "total_count",
    "hour",
    "day_of_week",
    "Precipitations",
    "HauteurNeige",
    "Temperature",
    "ForceVent",
    "month_of_year",
]


def nested_levels(rows: pd.DataFrame) -> list[pd.Series]:
    levels = [assign_cells(rows, SIZES[0])]
    for size in SIZES[1:]:
        levels.append(nest_keys(levels[-1], assign_cells(rows, size)))
    return levels


def stats_frame(predictions: np.ndarray, counts: np.ndarray, index: pd.Index) -> pd.DataFrame:
    columns = {f"mean_l{level + 1}": predictions[:, level] for level in range(len(SIZES))}
    columns.update({f"count_l{level + 1}": counts[:, level] for level in (2, 3)})
    return pd.DataFrame(columns, index=index)


def hour_level(levels: list[pd.Series], hours: pd.Series) -> pd.Series:
    """Key of the third level cell (1e-6) combined with the hour of the row."""
    return nest_keys(levels[2], hours.astype("int64"))


def hour_stats(
    codes: list[np.ndarray],
    hour_codes: np.ndarray,
    y_fit: np.ndarray,
    month: np.ndarray,
    n_fit: int,
    scheme: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shrunk mean and count of the cell by hour, with the cell's own mean as prior.

    Returns (fit_mean, fit_count, new_mean, new_count), each of shape (rows,).
    """
    chain = [*codes[:3], hour_codes]
    fit_pred, fit_count = fit_side_stats(chain, y_fit, month, n_fit, STRENGTH, scheme)
    new_pred, new_count = new_side_stats(chain, y_fit, n_fit, STRENGTH)
    return fit_pred[:, 3], fit_count[:, 3], new_pred[:, 3], new_count[:, 3]


def build_frames(
    X: pd.DataFrame,
    y: pd.Series,
    fit: pd.Series,
    val: pd.Series,
    scheme: str,
    hour_stat: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature frames (fit, validation) for the given row masks and out of fold scheme.

    With hour_stat, two extra columns hold the shrunk mean and the count of the 1e-6 cell by hour.
    """
    X_fit, y_fit, X_val, y_val = X[fit], y[fit], X[val], y[val]
    n_fit = len(X_fit)
    rows_all = pd.concat([X_fit, X_val])
    levels = nested_levels(rows_all)
    codes = to_codes(levels, n_fit)
    y_array = y_fit.to_numpy(dtype=float)
    month = X_fit["month_of_year"].to_numpy()
    fit_pred, fit_count = fit_side_stats(codes, y_array, month, n_fit, STRENGTH, scheme)
    val_pred, val_count = new_side_stats(codes, y_array, n_fit, STRENGTH)
    extra_fit = extra_val = None
    if hour_stat:
        hour_codes = to_codes([hour_level(levels, rows_all["hour"])], n_fit)[0]
        fm, fc, vm, vc = hour_stats(codes, hour_codes, y_array, month, n_fit, scheme)
        extra_fit = pd.DataFrame({"mean_l3h": fm, "count_l3h": fc}, index=X_fit.index)
        extra_val = pd.DataFrame({"mean_l3h": vm, "count_l3h": vc}, index=X_val.index)
    frames = []
    for rows, target, pred, count, extra in (
        (X_fit, y_fit, fit_pred, fit_count, extra_fit),
        (X_val, y_val, val_pred, val_count, extra_val),
    ):
        parts = [stats_frame(pred, count, rows.index)]
        if extra is not None:
            parts.append(extra)
        frame = pd.concat([*parts, rows[PASS_THROUGH], target.rename("y")], axis=1)
        floats = {c: "float32" for c in frame.columns if frame[c].dtype == "float64"}
        frames.append(frame.astype(floats))
    return frames[0], frames[1]
