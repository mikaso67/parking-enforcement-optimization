import numpy as np
import pandas as pd

CALENDAR_PERIODS = {
    "hour": 24.0,
    "day_of_week": 7.0,
    "month_of_year": 12.0,
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine and cosine cyclical encodings for calendar features.

    Adds sine and cosine representations for hour (period 24), day_of_week (period 7),
    and month_of_year (period 12).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'hour', 'day_of_week', and/or 'month_of_year'.

    Raises
    ------
    KeyError
        If one of the required columns is missing.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with added sine and cosine features.
    """
    missing = [col for col in CALENDAR_PERIODS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    res = df.copy()
    for col, period in CALENDAR_PERIODS.items():
        radians = 2.0 * np.pi * res[col] / period
        res[f"{col}_sin"] = np.sin(radians)
        res[f"{col}_cos"] = np.cos(radians)
    return res
