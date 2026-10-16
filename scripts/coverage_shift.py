"""Experiment 8: how familiar are the test cells, compared with the validation cells (seed 42).

Run from the repository root:
    PYTHONPATH=src python scripts/coverage_shift.py
The test rows are counted against all 12 training months (the final model's fit). Writes
data/interim/test_familiarity.json, read by scripts/error_analysis.py.
"""
import json
from pathlib import Path

import pandas as pd

from parking.baselines import to_markdown
from parking.data import load_test, load_train
from parking.error_analysis import familiarity_group, history_counts
from parking.oof_features import nested_levels


def finest_cell(rows: pd.DataFrame) -> pd.Series:
    return nested_levels(rows)[-1]


def shares(counts: pd.Series) -> dict[str, float]:
    return familiarity_group(counts).value_counts(normalize=True).sort_index().to_dict()


def main() -> None:
    X, _ = load_train(Path("data/raw"))
    fit_keys = finest_cell(X[X["month_of_year"] <= 9])
    val_keys = finest_cell(X[X["month_of_year"] >= 10])
    val_counts = pd.Series(history_counts(fit_keys, val_keys))
    del fit_keys, val_keys
    all_keys = finest_cell(X)
    X_test = load_test(Path("data/raw"))
    test_counts = pd.Series(history_counts(all_keys, finest_cell(X_test)))
    table = pd.DataFrame(
        {
            "validation (fit months 1-9)": shares(val_counts),
            "test (fit months 1-12)": shares(test_counts),
        }
    )
    print(to_markdown(table.round(3)))
    folder = Path("data/interim")
    folder.mkdir(parents=True, exist_ok=True)
    test_shares = {str(group): share for group, share in shares(test_counts).items()}
    (folder / "test_familiarity.json").write_text(json.dumps(test_shares))


if __name__ == "__main__":
    main()
