"""Experiment 8: where the final model ranks badly (expanding scheme), seed 42.

Run from the repository root:
    PYTHONPATH=src python scripts/error_analysis.py [fold]
Prints markdown tables and writes three figures to docs/figures/.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from train_lgbm import PARAMS  # noqa: E402

from parking.baselines import to_markdown  # noqa: E402
from parking.data import load_train  # noqa: E402
from parking.error_analysis import (  # noqa: E402
    BUCKET_LABELS,
    FAMILIARITY_LABELS,
    bucket_group,
    calibration_by_decile,
    extreme_share,
    familiarity_group,
    reweighted_spearman,
    spearman_by_group,
)
from parking.metrics import spearman  # noqa: E402
from parking.tuning import cached_frames  # noqa: E402

BLUE = "#1f5fbf"
GREY = "#8a8f98"
N_TREES = 100
FIGURES = Path("docs/figures")


def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def show(title: str, table, index: str) -> None:
    print(f"### {title}\n{to_markdown(table.set_index(index).round(3))}\n")


def bar_figure(table, title: str, xlabel: str, path: Path, reference: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(table["group"].astype(str), table["spearman"], color=BLUE, width=0.7)
    ax.axhline(reference, color=GREY, linestyle="--", linewidth=1)
    ax.text(len(table) - 0.5, reference, " overall", color=GREY, va="bottom", ha="right")
    style(ax, title, xlabel, "Spearman")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    fit, val = cached_frames(
        fold, "expanding", Path("data/interim"), lambda: load_train(Path("data/raw"))
    )
    features = [c for c in fit.columns if c not in ("y", "month_of_year")]
    model = lgb.LGBMRegressor(n_estimators=N_TREES, **PARAMS).fit(fit[features], fit["y"])
    pred = model.predict(val[features])
    y = val["y"]
    overall = spearman(y, pred)
    print(f"fold {fold}: {len(val)} validation rows, overall Spearman {overall:.3f}\n")

    buckets = bucket_group(val["total_count"])
    familiarity = familiarity_group(val["count_l4"])
    tables = {
        "by total_count bucket": spearman_by_group(buckets, y, pred),
        "by familiarity of the finest cell (fit rows in the cell)": spearman_by_group(
            familiarity, y, pred
        ),
        "by hour": spearman_by_group(val["hour"].astype(int), y, pred),
        "by day of week": spearman_by_group(val["day_of_week"].astype(int), y, pred),
    }
    for name, table in tables.items():
        show(f"Spearman {name}", table, "group")
    order = [f"{b} / {f}" for b in BUCKET_LABELS for f in FAMILIARITY_LABELS]
    labels = buckets.astype(str) + " / " + familiarity.astype(str)
    crossed = spearman_by_group(pd.Series(pd.Categorical(labels, categories=order)), y, pred)
    show("Spearman by bucket and familiarity", crossed, "group")
    show("Share of targets exactly 0 or 1", extreme_share(y, buckets), "group")
    calibration = calibration_by_decile(y, pred)
    show("Calibration by prediction decile", calibration, "bin")

    shares_path = Path("data/interim/test_familiarity.json")
    if shares_path.exists():
        test_shares = json.loads(shares_path.read_text())
        print("### Validation score if the cells were as familiar as the test cells")
        print(f"test shares: {test_shares}")
        score = reweighted_spearman(familiarity, y, pred, test_shares)
        print(f"reweighted Spearman: {score:.3f} (validation as is: {overall:.3f})\n")

    FIGURES.mkdir(parents=True, exist_ok=True)
    by_hour = tables["by hour"]
    bar_figure(by_hour, "Spearman by hour", "hour", FIGURES / "spearman_by_hour.png", overall)
    bar_figure(
        tables["by familiarity of the finest cell (fit rows in the cell)"],
        "Spearman by familiarity of the cell",
        "fit rows in the finest cell",
        FIGURES / "spearman_by_familiarity.png",
        overall,
    )
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(calibration["mean_pred"], calibration["mean_y"], marker="o", color=BLUE)
    low, high = float(np.min(calibration["mean_pred"])), float(np.max(calibration["mean_pred"]))
    ax.plot([low, high], [low, high], color=GREY, linestyle="--", linewidth=1)
    style(ax, "Calibration by prediction decile", "mean prediction", "mean invalid_ratio")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
