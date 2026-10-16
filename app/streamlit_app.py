"""Demo: where on-street parking checks are most useful, by day and time of day.

Run from the repository root after scripts/final_predict.py and scripts/priority_map.py:
    PYTHONPATH=src streamlit run app/streamlit_app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from parking.priority import DAY_TYPES, SLOT_LABELS, select_slot
from parking.style import BACKGROUND, MUTED, OUTLINE, RATE_MAP, padded_limits, style_axes

TABLE_PATH = Path("data/processed/priority_cells.csv")


@st.cache_data
def load_table() -> pd.DataFrame:
    return pd.read_csv(TABLE_PATH)


def draw_map(part: pd.DataFrame, top: pd.DataFrame, limits: tuple[float, float, float, float]):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BACKGROUND)
    ordered = part.sort_values("mean_prediction")
    points = ax.scatter(
        ordered["longitude"], ordered["latitude"], c=ordered["mean_prediction"],
        cmap=RATE_MAP, s=6, linewidths=0,
    )
    ax.scatter(top["longitude"], top["latitude"], s=22, facecolors="none",
               edgecolors=OUTLINE, linewidths=0.5)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Scaled longitude")
    ax.set_ylabel("Scaled latitude")
    style_axes(ax)
    bar = fig.colorbar(points, ax=ax, shrink=0.8)
    bar.set_label("Expected infringement rate", color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_edgecolor(BACKGROUND)
    return fig


def explain() -> None:
    st.write(
        "Cars with cameras check parked vehicles on the street, and some places collect far more "
        "infringements than others. A model learned where, from a year of past checks. This page "
        "shows the result: small areas of the city, ranked by the infringement rate the model "
        "expects for the day and time you pick. The cars should go to the circled areas first."
    )
    with st.expander("How to read the map"):
        st.markdown(
            "- Each dot is a small square area of the city. Positions are scaled, so there are no "
            "street names.\n"
            "- The colour is the rate the model expects there for the chosen day and time. "
            "Brighter means more infringements expected.\n"
            "- Circled dots are the areas flagged as priority. You choose how many on the left.\n"
            "- An area backed by only a few past checks is noisy. Raise the minimum evidence to "
            "keep only well covered areas.\n"
            "- This is a ranking, not a measured rate: use it to decide where to go first."
        )


def main() -> None:
    st.set_page_config(page_title="Where to check parking first", layout="wide")
    st.title("Where to check parking first")
    explain()
    if not TABLE_PATH.exists():
        st.error("Table not found. Run scripts/final_predict.py then scripts/priority_map.py.")
        return
    table = load_table()
    day = st.sidebar.radio("Day", DAY_TYPES)
    slot = st.sidebar.selectbox("Time of day", SLOT_LABELS)
    share = st.sidebar.slider("Share of areas to flag", 0.02, 0.30, 0.10, 0.01)
    min_rows = st.sidebar.slider("Minimum evidence per area", 20, 200, 20, 10,
                                 help="Number of predictions behind each area.")
    part, top = select_slot(table, day, slot, share, min_rows)
    if part.empty:
        st.warning("No area matches these settings.")
        return
    left, middle, right = st.columns(3)
    left.metric("Areas on the map", f"{len(part):,}")
    middle.metric("Areas flagged", f"{len(top):,}")
    right.metric("Expected rate, flagged vs all", f"{top['mean_prediction'].mean():.2f} vs "
                 f"{part['mean_prediction'].mean():.2f}")
    limits = (*padded_limits(table["longitude"]), *padded_limits(table["latitude"]))
    figure = draw_map(part, top, limits)
    st.pyplot(figure)
    plt.close(figure)
    st.subheader("The 20 areas to check first")
    columns = ["longitude", "latitude", "mean_prediction", "rows"]
    best = top.sort_values("mean_prediction", ascending=False).head(20)[columns]
    best = best.rename(columns={
        "longitude": "Longitude (scaled)", "latitude": "Latitude (scaled)",
        "mean_prediction": "Expected rate", "rows": "Evidence",
    })
    st.dataframe(best.round(6), hide_index=True)
    st.download_button("Download the flagged areas (CSV)", top[columns].to_csv(index=False),
                       file_name="flagged_areas.csv", mime="text/csv")


main()
