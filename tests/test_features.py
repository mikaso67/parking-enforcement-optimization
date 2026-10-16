import numpy as np
import pandas as pd
import pytest

from parking.features import add_calendar_features


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "hour": [0, 6, 12, 18, 23, 24],
            "day_of_week": [0, 1, 2, 3, 4, 7],
            "month_of_year": [0, 1, 6, 11, 12, 3],
            "other_col": [10, 20, 30, 40, 50, 60],
        }
    )


def test_add_calendar_features_periodicity(sample_df):
    res = add_calendar_features(sample_df)

    # hour 0 and hour 24
    assert np.isclose(res.loc[0, "hour_sin"], res.loc[5, "hour_sin"])
    assert np.isclose(res.loc[0, "hour_cos"], res.loc[5, "hour_cos"])

    # day_of_week 0 and 7
    assert np.isclose(res.loc[0, "day_of_week_sin"], res.loc[5, "day_of_week_sin"])
    assert np.isclose(res.loc[0, "day_of_week_cos"], res.loc[5, "day_of_week_cos"])

    # month_of_year 0 and 12
    assert np.isclose(res.loc[0, "month_of_year_sin"], res.loc[4, "month_of_year_sin"])
    assert np.isclose(res.loc[0, "month_of_year_cos"], res.loc[4, "month_of_year_cos"])


def test_add_calendar_features_known_values():
    df = pd.DataFrame(
        {
            "hour": [0, 6],
            "day_of_week": [0, 0],
            "month_of_year": [0, 0],
        }
    )
    res = add_calendar_features(df)

    # hour 0: sin = 0, cos = 1
    assert np.isclose(res.loc[0, "hour_sin"], 0.0)
    assert np.isclose(res.loc[0, "hour_cos"], 1.0)

    # hour 6: sin = 1, cos = 0
    assert np.isclose(res.loc[1, "hour_sin"], 1.0)
    assert np.isclose(res.loc[1, "hour_cos"], 0.0, atol=1e-7)


def test_add_calendar_features_bounds(sample_df):
    res = add_calendar_features(sample_df)
    encoded_cols = [
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_of_year_sin",
        "month_of_year_cos",
    ]
    for col in encoded_cols:
        assert (res[col] >= -1.0 - 1e-7).all()
        assert (res[col] <= 1.0 + 1e-7).all()


def test_add_calendar_features_no_missing_values(sample_df):
    res = add_calendar_features(sample_df)
    assert not res.isna().any().any()


def test_add_calendar_features_input_untouched(sample_df):
    original_df = sample_df.copy()
    _ = add_calendar_features(sample_df)
    pd.testing.assert_frame_equal(sample_df, original_df)


def test_add_calendar_features_missing_columns_raise(sample_df):
    incomplete = sample_df.drop(columns=["day_of_week", "month_of_year"])
    with pytest.raises(KeyError, match="day_of_week") as excinfo:
        add_calendar_features(incomplete)
    assert "month_of_year" in str(excinfo.value)
