import numpy as np
import pandas as pd

CELL_BASE = 10_000_003
MAX_CELL_INDEX = 5_000_000


def assign_cells(
    df: pd.DataFrame,
    cell_size: float,
    lon_col: str = "longitude_scaled",
    lat_col: str = "latitude_scaled",
) -> pd.Series:
    """Assign 2D grid cell IDs to coordinates in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing longitude and latitude columns.
    cell_size : float
        Grid cell side length. Must be greater than 0.
    lon_col : str, default "longitude_scaled"
        Column name for longitude.
    lat_col : str, default "latitude_scaled"
        Column name for latitude.

    Notes
    -----
    Cell indexes are floor(coordinate / cell_size) in float64. A coordinate that is an exact
    decimal multiple of cell_size (0.3 with 0.1) can fall in the lower cell because of binary
    rounding. The formula is kept as is so that results stay reproducible.

    Returns
    -------
    pd.Series
        Series of int64 grid cell IDs with the same index as df.

    Raises
    ------
    ValueError
        If cell_size <= 0, if coordinates contain missing values (NaN), or if cell_size is so
        small that distinct cells would share an id.
    KeyError
        If lon_col or lat_col is absent from df.
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")

    missing_cols = [col for col in (lon_col, lat_col) if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing column(s): {missing_cols}")

    lon_series = df[lon_col]
    lat_series = df[lat_col]

    if lon_series.isna().any() or lat_series.isna().any():
        raise ValueError("Coordinates contain missing values (NaN).")

    lon_arr = lon_series.to_numpy(dtype=np.float64)
    lat_arr = lat_series.to_numpy(dtype=np.float64)

    cs = np.float64(cell_size)
    ix_float = np.floor(lon_arr / cs)
    iy_float = np.floor(lat_arr / cs)
    largest = max(np.abs(ix_float).max(initial=0.0), np.abs(iy_float).max(initial=0.0))
    if not np.isfinite(largest) or largest >= MAX_CELL_INDEX:
        raise ValueError(f"cell_size {cell_size} is too small: cell indexes would collide")

    cell_ids = ix_float.astype(np.int64) * CELL_BASE + iy_float.astype(np.int64)

    return pd.Series(cell_ids, index=df.index, dtype="int64")


def point_keys(
    df: pd.DataFrame,
    lon_col: str = "longitude_scaled",
    lat_col: str = "latitude_scaled",
) -> pd.Series:
    """Return one int64 key per exact coordinate pair (float32 bit patterns), no grid."""
    missing_cols = [col for col in (lon_col, lat_col) if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing column(s): {missing_cols}")
    lon_bits = df[lon_col].to_numpy(dtype=np.float32).view(np.uint32).astype(np.int64)
    lat_bits = df[lat_col].to_numpy(dtype=np.float32).view(np.uint32).astype(np.int64)
    return pd.Series((lon_bits << 32) | lat_bits, index=df.index, dtype="int64")


HASH_PARENT = np.uint64(0x9E3779B97F4A7C15)
HASH_CHILD = np.uint64(0xC2B2AE3D27D4EB4F)
HASH_MIX = np.uint64(0xFF51AFD7ED558CCD)


def nest_keys(parent: pd.Series, child: pd.Series) -> pd.Series:
    """Return one int64 key per (parent key, child key) pair, so a child cell has one parent.

    Grid cells of sizes that are not exact multiples can straddle two parent cells. Keying the
    child by the pair keeps the hierarchy a tree. The key is a fixed 64 bit hash of the pair, so
    it depends on the two inputs only and not on which other rows are present.
    """
    if not parent.index.equals(child.index):
        raise ValueError("parent and child keys must share the same index")
    with np.errstate(over="ignore"):
        mixed = parent.to_numpy().astype(np.uint64) * HASH_PARENT
        mixed = (mixed ^ (mixed >> np.uint64(32))) + child.to_numpy().astype(np.uint64) * HASH_CHILD
        mixed = (mixed ^ (mixed >> np.uint64(29))) * HASH_MIX
        mixed = mixed ^ (mixed >> np.uint64(32))
    return pd.Series(mixed.view(np.int64), index=child.index, dtype="int64")
