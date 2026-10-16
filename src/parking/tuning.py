"""Rolling origin folds and the helpers used by the tuning scripts (D5, D9)."""
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from parking.oof_features import build_frames

# Folds 1 to 3 roll forward in time. Fold 4 is a season check: it validates on the months of the
# test set (1 to 5) and fits on the other months, so the fit part is not earlier than the
# validation part.
SEASON_CHECK_FOLD = 4
FOLDS = {1: ((1, 3), (4, 6)), 2: ((1, 6), (7, 9)), 3: ((1, 9), (10, 12)), 4: ((6, 12), (1, 5))}


def warn_if_season_check(fold: int) -> None:
    if fold == SEASON_CHECK_FOLD:
        warnings.warn(
            "Fold 4 fits on months later than part of its validation months: it is a season "
            "check, not a forward-time validation. Do not use it to choose a model or settings.",
            UserWarning,
            stacklevel=3,
        )


def fold_masks(month: pd.Series, fold: int) -> tuple[pd.Series, pd.Series]:
    """Boolean masks of the fit rows and the validation rows of a fold.

    Folds 1 to 3 are forward-time validations. Fold 4 is a season check, a comparison and not a
    validation: its fit months are later than part of its validation months, so it warns.
    """
    warn_if_season_check(fold)
    (fit_lo, fit_hi), (val_lo, val_hi) = FOLDS[fold]
    return month.between(fit_lo, fit_hi), month.between(val_lo, val_hi)


def parse_overrides(args: list[str], integer_names: set[str]) -> dict[str, int | float]:
    """Turn ['num_leaves=15', 'learning_rate=0.03'] into typed values."""
    overrides: dict[str, int | float] = {}
    for item in args:
        name, value = item.split("=")
        overrides[name] = int(value) if name in integer_names else float(value)
    return overrides


def cached_frames(
    fold: int,
    scheme: str,
    folder: Path,
    load: Callable[[], tuple[pd.DataFrame, pd.Series]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature frames of a fold, read from folder when cached, otherwise built and written.

    Fold 4 warns even when its frames come from the cache (see fold_masks).
    """
    warn_if_season_check(fold)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"fold{fold}_{scheme}_{part}.parquet" for part in ("fit", "val")]
    if not all(path.exists() for path in paths):
        X, y = load()
        fit, val = fold_masks(X["month_of_year"], fold)
        fit_frame, val_frame = build_frames(X, y, fit, val, scheme)
        fit_frame.to_parquet(paths[0])
        val_frame.to_parquet(paths[1])
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
