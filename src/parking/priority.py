"""Priority zones: cells ranked by predicted infringement rate for a time slot."""
import numpy as np
import pandas as pd

SLOT_LABELS = ["Morning (7h to 10h)", "Midday (11h to 14h)", "Afternoon (15h to 17h)"]
SLOT_EDGES = [6, 10, 14, 17]
DAY_TYPES = ["Weekday", "Saturday"]


def time_slot(hours: pd.Series) -> pd.Series:
    """Slot of each row (7h to 10h, 11h to 14h, 15h to 17h); other hours are missing."""
    return pd.cut(hours, bins=SLOT_EDGES, labels=SLOT_LABELS, right=True)


def day_type(day_of_week: pd.Series) -> pd.Series:
    """Weekday for Monday to Friday (1 to 5), Saturday for 6; other values are missing."""
    labels = pd.Series(pd.NA, index=day_of_week.index, dtype="object")
    labels[day_of_week.between(1, 5)] = DAY_TYPES[0]
    labels[day_of_week == 6] = DAY_TYPES[1]
    return labels


def inside_city(rows: pd.DataFrame, spread_factor: float = 10.0) -> pd.Series:
    """True for rows whose position is close to the bulk of the points.

    The bulk is the range between the 1% and 99% quantiles of each coordinate. A point farther
    than spread_factor times that range from the median is treated as an invalid GPS position.
    """
    inside = pd.Series(True, index=rows.index)
    for column in ("longitude_scaled", "latitude_scaled"):
        values = rows[column].astype("float64")
        low, median, high = values.quantile([0.01, 0.5, 0.99])
        inside &= (values - median).abs() <= spread_factor * (high - low)
    return inside


def priority_cells(rows: pd.DataFrame, size: float, min_rows: int) -> pd.DataFrame:
    """Mean predicted rate per grid cell and time slot, keeping cells with enough rows.

    rows needs the columns longitude_scaled, latitude_scaled, hour, day_of_week and prediction.
    Rows outside the three slots, on another day than Monday to Saturday, or at an invalid GPS
    position (see inside_city) are ignored.
    The result has one line per (day type, slot, cell) with the cell centre, the mean prediction
    and the number of rows, sorted by decreasing prediction inside each day type and slot.
    """
    rows = rows[inside_city(rows)]
    keys = pd.DataFrame(
        {
            "day_type": day_type(rows["day_of_week"]),
            "slot": time_slot(rows["hour"]),
            "cell_x": np.floor(rows["longitude_scaled"].astype("float64") / size).astype("int64"),
            "cell_y": np.floor(rows["latitude_scaled"].astype("float64") / size).astype("int64"),
            "prediction": rows["prediction"],
        }
    )
    grouped = keys.groupby(["day_type", "slot", "cell_x", "cell_y"], observed=True)["prediction"]
    table = grouped.agg(mean_prediction="mean", rows="size").reset_index()
    table = table[table["rows"] >= min_rows].copy()
    table["longitude"] = (table["cell_x"] + 0.5) * size
    table["latitude"] = (table["cell_y"] + 0.5) * size
    order = ["day_type", "slot", "mean_prediction"]
    return table.sort_values(order, ascending=[True, True, False]).reset_index(drop=True)


def top_cells(table: pd.DataFrame, share: float) -> pd.DataFrame:
    """The best cells of each day type and slot: ceil(share x cells), at least one."""
    groups = table.groupby(["day_type", "slot"], observed=True)["mean_prediction"]
    rank = groups.rank(ascending=False, method="first")
    keep = np.ceil(groups.transform("size") * share)
    return table[rank <= keep]


def select_slot(
    table: pd.DataFrame, day: str, slot: str, share: float, min_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cells of one day type and slot with at least min_rows rows, and their top share."""
    part = table[(table["day_type"] == day) & (table["slot"] == slot) & (table["rows"] >= min_rows)]
    return part, top_cells(part, share) if len(part) else part
