import numpy as np
import pandas as pd

from parking.priority import (
    SLOT_LABELS,
    day_type,
    inside_city,
    priority_cells,
    select_slot,
    time_slot,
    top_cells,
)


def rows():
    return pd.DataFrame(
        {
            "longitude_scaled": [0.10001, 0.10002, 0.10001, 0.20001, 0.20001],
            "latitude_scaled": [0.30001, 0.30002, 0.30001, 0.40001, 0.40001],
            "hour": [8, 8, 12, 8, 16],
            "day_of_week": [1, 2, 3, 1, 6],
            "prediction": [0.2, 0.4, 0.9, 0.7, 0.1],
        }
    )


def test_time_slot_boundaries():
    slots = time_slot(pd.Series([7, 10, 11, 14, 15, 17]))
    assert list(slots) == [SLOT_LABELS[0]] * 2 + [SLOT_LABELS[1]] * 2 + [SLOT_LABELS[2]] * 2


def test_hours_outside_the_slots_are_missing():
    assert time_slot(pd.Series([3, 6, 18, 22])).isna().all()


def test_day_type_groups_monday_to_friday_and_saturday():
    labels = day_type(pd.Series([1, 5, 6, 0, 7]))
    assert list(labels[:3]) == ["Weekday", "Weekday", "Saturday"]
    assert labels[3:].isna().all()


def test_priority_cells_ignore_hours_outside_the_slots():
    data = rows()
    data.loc[0, "hour"] = 3
    table = priority_cells(data, size=1e-3, min_rows=1)
    assert table["rows"].sum() == 4


def test_priority_cells_separate_weekday_and_saturday():
    table = priority_cells(rows(), size=1e-3, min_rows=1)
    saturday = table[table["day_type"] == "Saturday"]
    assert len(saturday) == 1 and abs(saturday.iloc[0]["mean_prediction"] - 0.1) < 1e-9


def test_priority_cells_averages_inside_cell_and_slot():
    table = priority_cells(rows(), size=1e-3, min_rows=1)
    morning = table[table["slot"] == SLOT_LABELS[0]]
    first = morning[morning["cell_x"] == 100].iloc[0]
    assert first["rows"] == 2
    assert abs(first["mean_prediction"] - 0.3) < 1e-9


def test_priority_cells_drops_small_cells_and_sorts_descending():
    table = priority_cells(rows(), size=1e-3, min_rows=2)
    assert list(table["rows"]) == [2]
    full = priority_cells(rows(), size=1e-3, min_rows=1)
    morning = full[full["slot"] == SLOT_LABELS[0]]["mean_prediction"].tolist()
    assert morning == sorted(morning, reverse=True)


def test_top_cells_keeps_the_best_share_per_slot():
    table = priority_cells(rows(), size=1e-3, min_rows=1)
    top = top_cells(table, share=0.5)
    morning = top[(top["slot"] == SLOT_LABELS[0]) & (top["day_type"] == "Weekday")]
    assert len(morning) == 1
    assert abs(morning.iloc[0]["mean_prediction"] - 0.7) < 1e-9


def test_top_cells_keeps_at_least_one_cell_for_a_small_share():
    table = priority_cells(rows(), size=1e-3, min_rows=1)
    top = top_cells(table, share=0.01)
    assert len(top) == table.groupby(["day_type", "slot"], observed=True).ngroups


def test_float32_coordinates_use_the_float64_grid():
    data = rows()
    narrow = data.assign(
        longitude_scaled=data["longitude_scaled"].astype("float32"),
        latitude_scaled=data["latitude_scaled"].astype("float32"),
    )
    wide = narrow.astype({"longitude_scaled": "float64", "latitude_scaled": "float64"})
    a = priority_cells(narrow, size=1e-3, min_rows=1)
    b = priority_cells(wide, size=1e-3, min_rows=1)
    assert a[["cell_x", "cell_y"]].equals(b[["cell_x", "cell_y"]])


def test_select_slot_filters_and_returns_the_top_cells():
    table = priority_cells(rows(), size=1e-3, min_rows=1)
    part, top = select_slot(table, "Weekday", SLOT_LABELS[0], share=0.5, min_rows=1)
    assert len(part) == 2 and len(top) == 1
    only_big, _ = select_slot(table, "Weekday", SLOT_LABELS[0], share=0.5, min_rows=2)
    assert len(only_big) == 1


def test_select_slot_handles_an_empty_selection():
    table = priority_cells(rows(), size=1e-3, min_rows=1)
    part, top = select_slot(table, "Saturday", SLOT_LABELS[0], share=0.5, min_rows=1)
    assert part.empty and top.empty


def test_inside_city_flags_points_far_from_the_bulk():
    rng = np.random.default_rng(42)
    n = 1000
    data = pd.DataFrame(
        {
            "longitude_scaled": 0.9988 + rng.normal(0, 1e-4, n),
            "latitude_scaled": 0.9961 + rng.normal(0, 1e-4, n),
        }
    )
    data.loc[0, "latitude_scaled"] = 0.0
    data.loc[1, "longitude_scaled"] = 0.9808
    inside = inside_city(data)
    assert not inside[0] and not inside[1]
    assert inside[2:].all()


def test_priority_cells_drop_invalid_gps_positions():
    rng = np.random.default_rng(42)
    n = 500
    data = pd.DataFrame(
        {
            "longitude_scaled": 0.9988 + rng.normal(0, 1e-4, n),
            "latitude_scaled": 0.9961 + rng.normal(0, 1e-4, n),
            "hour": 8,
            "day_of_week": 2,
            "prediction": 0.5,
        }
    )
    data.loc[0, ["longitude_scaled", "latitude_scaled", "prediction"]] = [0.9808, 0.0, 1.0]
    table = priority_cells(data, size=1e-4, min_rows=1)
    assert table["rows"].sum() == n - 1
    assert table["mean_prediction"].max() < 1.0
