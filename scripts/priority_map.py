"""Map of priority zones by time slot, from the test predictions.

Run from the repository root (after final_predict.py):
    PYTHONPATH=src python scripts/priority_map.py
Writes docs/figures/priority_zones_weekday.png, docs/figures/priority_zones_saturday.png and
data/processed/priority_cells.csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from parking.priority import DAY_TYPES, SLOT_LABELS, priority_cells, top_cells  # noqa: E402
from parking.style import (  # noqa: E402
    BACKGROUND,
    MUTED,
    OUTLINE,
    RATE_MAP,
    padded_limits,
    style_axes,
)

CELL_SIZE = 1e-5
MIN_ROWS = 20
TOP_SHARE = 0.10


def draw(table: pd.DataFrame, day: str, out: Path) -> None:
    """One figure for a day type; colour scale and axes come from the whole table."""
    low, high = table["mean_prediction"].quantile([0.01, 0.99])
    x_min, x_max = padded_limits(table["longitude"])
    y_min, y_max = padded_limits(table["latitude"])
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.6), sharex=True, sharey=True, facecolor=BACKGROUND
    )
    for ax, label in zip(axes, SLOT_LABELS, strict=True):
        part = table[(table["day_type"] == day) & (table["slot"] == label)]
        part = part.sort_values("mean_prediction")
        top = top_cells(part, TOP_SHARE)
        points = ax.scatter(
            part["longitude"], part["latitude"], c=part["mean_prediction"],
            cmap=RATE_MAP, vmin=low, vmax=high, s=4, linewidths=0,
        )
        ax.scatter(
            top["longitude"], top["latitude"], s=14, facecolors="none",
            edgecolors=OUTLINE, linewidths=0.4,
        )
        ax.set_title(
            f"{day}, {label}\n{len(part):,} areas, top {TOP_SHARE:.0%} circled", fontsize=10
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        style_axes(ax)
    bar = fig.colorbar(points, ax=axes, shrink=0.8)
    bar.set_label("Expected infringement rate", color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_edgecolor(BACKGROUND)
    fig.supxlabel("Scaled longitude", fontsize=9, color=MUTED)
    fig.supylabel("Scaled latitude", fontsize=9, color=MUTED)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BACKGROUND)


def main() -> None:
    rows = pd.read_parquet("data/processed/test_features.parquet")
    table = priority_cells(rows, CELL_SIZE, MIN_ROWS)
    table.to_csv("data/processed/priority_cells.csv", index=False)
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    for day in DAY_TYPES:
        draw(table, day, out / f"priority_zones_{day.lower()}.png")
    summary = table.groupby(["day_type", "slot"], observed=True)["mean_prediction"].describe()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
