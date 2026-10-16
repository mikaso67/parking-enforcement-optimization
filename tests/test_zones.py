import numpy as np
import pandas as pd
import pytest

from parking.zones import assign_cells, nest_keys, point_keys


def test_close_and_far_points():
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.01, 0.02, 0.85],
            "latitude_scaled": [0.01, 0.02, 0.85],
        }
    )
    res = assign_cells(df, cell_size=0.1)

    # Close points (0.01, 0.01) and (0.02, 0.02) share cell (0, 0)
    assert res.iloc[0] == res.iloc[1]
    # Far point (0.85, 0.85) has a different cell id
    assert res.iloc[0] != res.iloc[2]


def test_cell_size_respected_small_move():
    # cell_size = 0.05
    # Move within cell: (0.01 -> 0.03) stays in cell 0
    # Move across boundary: (0.01 -> 0.06) crosses boundary into cell 1
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.01, 0.03, 0.06],
            "latitude_scaled": [0.01, 0.01, 0.01],
        }
    )
    res = assign_cells(df, cell_size=0.05)

    assert res.iloc[0] == res.iloc[1]
    assert res.iloc[0] != res.iloc[2]


def test_index_preserved():
    idx = ["row_a", "row_b", "row_c"]
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.1, 0.2, 0.3],
            "latitude_scaled": [0.4, 0.5, 0.6],
        },
        index=idx,
    )
    res = assign_cells(df, cell_size=0.1)

    assert list(res.index) == idx
    assert res.dtype == np.dtype("int64")


def test_input_untouched():
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.1, 0.2],
            "latitude_scaled": [0.3, 0.4],
        }
    )
    original_df = df.copy()
    _ = assign_cells(df, cell_size=0.1)

    pd.testing.assert_frame_equal(df, original_df)


def test_coordinate_on_boundary_goes_to_upper_cell():
    # For cell_size = 0.1:
    # x in [0.0, 0.1) is ix = 0 (cell 0)
    # x = 0.1 is exactly on the boundary and goes to ix = 1 (upper cell)
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.05, 0.10],
            "latitude_scaled": [0.05, 0.10],
        }
    )
    res = assign_cells(df, cell_size=0.1)

    ix_0 = 0
    iy_0 = 0
    cell_id_0 = ix_0 * 10_000_003 + iy_0

    ix_1 = 1
    iy_1 = 1
    cell_id_1 = ix_1 * 10_000_003 + iy_1

    assert res.iloc[0] == cell_id_0
    assert res.iloc[1] == cell_id_1


def test_error_cell_size_zero_or_negative():
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.1],
            "latitude_scaled": [0.2],
        }
    )
    with pytest.raises(ValueError, match="cell_size"):
        assign_cells(df, cell_size=0.0)

    with pytest.raises(ValueError, match="cell_size"):
        assign_cells(df, cell_size=-0.05)


def test_error_missing_coordinate_nan():
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.1, np.nan],
            "latitude_scaled": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="missing values"):
        assign_cells(df, cell_size=0.1)


def test_error_missing_columns():
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.1, 0.2],
        }
    )
    with pytest.raises(KeyError, match="latitude_scaled"):
        assign_cells(df, cell_size=0.1)

    df_none = pd.DataFrame({"other_col": [1, 2]})
    with pytest.raises(KeyError) as excinfo:
        assign_cells(df_none, cell_size=0.1)
    assert "longitude_scaled" in str(excinfo.value)
    assert "latitude_scaled" in str(excinfo.value)


def test_small_cell_size_near_boundary_point_spacing():
    # cell_size = 1e-5, coordinates near 0.9988, points 2e-5 apart get distinct ids
    lon1 = 0.9988
    lon2 = lon1 + 2e-5
    df = pd.DataFrame(
        {
            "longitude_scaled": [lon1, lon2],
            "latitude_scaled": [0.5, 0.5],
        }
    )
    res = assign_cells(df, cell_size=1e-5)

    assert res.iloc[0] != res.iloc[1]


def test_assign_cells_rejects_cell_size_that_would_collide() -> None:
    df = pd.DataFrame({"longitude_scaled": [0.0, 1e-8], "latitude_scaled": [0.1, 0.0]})
    with pytest.raises(ValueError, match="too small"):
        assign_cells(df, 1e-8)


def test_assign_cells_rejects_cell_size_that_overflows_int64() -> None:
    df = pd.DataFrame({"longitude_scaled": [0.9988], "latitude_scaled": [0.9961]})
    with pytest.raises(ValueError, match="too small"):
        assign_cells(df, 1e-20)


def test_point_keys_identical_points_share_a_key_and_others_do_not() -> None:
    df = pd.DataFrame(
        {
            "longitude_scaled": [0.9988, 0.9988, 0.9988, 0.99881],
            "latitude_scaled": [0.9961, 0.9961, 0.99611, 0.9961],
        },
        index=[5, 6, 7, 8],
    )
    keys = point_keys(df)
    assert keys.dtype == np.int64
    assert list(keys.index) == [5, 6, 7, 8]
    assert keys.iloc[0] == keys.iloc[1]
    assert keys.nunique() == 3


def test_point_keys_missing_column_raises() -> None:
    with pytest.raises(KeyError, match="latitude_scaled"):
        point_keys(pd.DataFrame({"longitude_scaled": [0.1]}))


def test_nest_keys_splits_a_child_cell_that_straddles_two_parents() -> None:
    parent = pd.Series([1, 2, 2])
    child = pd.Series([7, 7, 7])
    nested = nest_keys(parent, child)
    assert nested.iloc[0] != nested.iloc[1]
    assert nested.iloc[1] == nested.iloc[2]
    assert nested.dtype == np.int64


def test_nest_keys_same_pair_gets_the_same_code_and_index_is_kept() -> None:
    parent = pd.Series([1, 1, 3, 1], index=[10, 20, 30, 40])
    child = pd.Series([5, 5, 5, 6], index=[10, 20, 30, 40])
    nested = nest_keys(parent, child)
    assert list(nested.index) == [10, 20, 30, 40]
    assert nested.loc[10] == nested.loc[20]
    assert nested.nunique() == 3


def test_nest_keys_rejects_mismatched_indexes() -> None:
    with pytest.raises(ValueError, match="same index"):
        nest_keys(pd.Series([1], index=[0]), pd.Series([1], index=[1]))


def test_nest_keys_does_not_depend_on_the_other_rows() -> None:
    parent = pd.Series([1, 2, 2, 9])
    child = pd.Series([7, 7, 8, 7])
    full = nest_keys(parent, child)
    alone = nest_keys(parent.iloc[:2], child.iloc[:2])
    assert (full.iloc[:2].to_numpy() == alone.to_numpy()).all()


def test_nest_keys_has_no_collision_on_a_dense_grid_of_pairs() -> None:
    parent = pd.Series(np.repeat(np.arange(300), 300))
    child = pd.Series(np.tile(np.arange(300), 300))
    assert nest_keys(parent, child).nunique() == 300 * 300
