from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from parking.priority import SLOT_LABELS  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def synthetic_table() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame(
        {
            "day_type": "Weekday",
            "slot": SLOT_LABELS[0],
            "cell_x": np.arange(n),
            "cell_y": np.arange(n),
            "mean_prediction": rng.uniform(0.1, 0.9, n),
            "rows": rng.integers(20, 60, n),
            "longitude": 0.9988 + np.arange(n) * 1e-5,
            "latitude": 0.9960 + np.arange(n) * 1e-5,
        }
    )


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_table(folder: Path) -> None:
    target = folder / "data" / "processed"
    target.mkdir(parents=True)
    synthetic_table().to_csv(target / "priority_cells.csv", index=False)


def test_app_shows_metrics_for_a_valid_selection(workdir) -> None:
    write_table(workdir)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert labels == ["Areas on the map", "Areas flagged", "Expected rate, flagged vs all"]
    assert app.metric[0].value == "40"


def test_app_warns_when_no_cell_matches(workdir) -> None:
    write_table(workdir)
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.slider[1].set_value(200).run()
    assert not app.exception
    assert [warning.value for warning in app.warning] == ["No area matches these settings."]
    app.sidebar.radio[0].set_value("Saturday").run()
    assert not app.exception and len(app.warning) == 1


def test_app_explains_a_missing_table(workdir) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert len(app.error) == 1 and "Table not found" in app.error[0].value
