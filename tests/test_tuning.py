import warnings

import numpy as np
import pandas as pd
import pytest
from test_oof_features import synthetic

from parking.tuning import FOLDS, cached_frames, fold_masks, parse_overrides


def test_parse_overrides_converts_types() -> None:
    result = parse_overrides(["num_leaves=15", "learning_rate=0.03"], {"num_leaves"})
    assert result == {"num_leaves": 15, "learning_rate": 0.03}
    assert isinstance(result["num_leaves"], int)
    assert isinstance(result["learning_rate"], float)


def test_parse_overrides_empty() -> None:
    assert parse_overrides([], {"num_leaves"}) == {}


@pytest.mark.parametrize("fold", sorted(FOLDS))
def test_fold_masks_follow_the_fold_definition(fold: int) -> None:
    month = pd.Series(np.arange(1, 13))
    fit, val = fold_masks(month, fold)
    (fit_lo, fit_hi), (val_lo, val_hi) = FOLDS[fold]
    assert month[fit].min() == fit_lo and month[fit].max() == fit_hi
    assert month[val].min() == val_lo and month[val].max() == val_hi
    assert not (fit & val).any()


def test_season_check_fold_warns_and_forward_folds_do_not() -> None:
    month = pd.Series(np.arange(1, 13))
    with pytest.warns(UserWarning, match="season check"):
        fold_masks(month, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fold in (1, 2, 3):
            fold_masks(month, fold)


def test_cached_season_check_frames_still_warn(tmp_path) -> None:
    X, y = synthetic(n=900)
    X["month_of_year"] = np.tile(np.arange(1, 13), 75)
    folder = tmp_path / "interim"
    with pytest.warns(UserWarning, match="season check"):
        cached_frames(4, "expanding", folder, lambda: (X, y))
    with pytest.warns(UserWarning, match="season check"):
        cached_frames(4, "expanding", folder, lambda: (X, y))


def test_cached_frames_builds_then_reads_the_cache(tmp_path) -> None:
    X, y = synthetic(n=900)
    X["month_of_year"] = np.tile(np.arange(1, 7), 150)
    calls = []

    def load() -> tuple[pd.DataFrame, pd.Series]:
        calls.append(1)
        return X, y

    folder = tmp_path / "missing" / "interim"
    fit_a, val_a = cached_frames(1, "expanding", folder, load)
    fit_b, val_b = cached_frames(1, "expanding", folder, load)
    assert len(calls) == 1
    assert len(fit_a) == (X["month_of_year"] <= 3).sum()
    assert len(val_a) == X["month_of_year"].between(4, 6).sum()
    assert np.allclose(fit_a["mean_l4"], fit_b["mean_l4"])
    assert np.allclose(val_a["mean_l4"], val_b["mean_l4"])
