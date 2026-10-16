import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from parking.style import (  # noqa: E402
    ACCENT,
    BACKGROUND,
    GRID,
    RATE_MAP,
    padded_limits,
    style_axes,
)


def test_rate_map_goes_from_dark_to_light():
    low, high = RATE_MAP(0.0), RATE_MAP(1.0)
    assert sum(low[:3]) < sum(high[:3])


def test_style_axes_uses_the_shared_colours():
    fig, ax = plt.subplots()
    style_axes(ax)
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_edgecolor()[:3] == matplotlib.colors.to_rgb(GRID)
    plt.close(fig)


def test_palette_has_one_accent_blue():
    assert ACCENT.startswith("#") and BACKGROUND != ACCENT


def test_padded_limits_cover_the_values_with_a_margin():
    import pandas as pd

    low, high = padded_limits(pd.Series([1.0, 3.0]), padding=0.1)
    assert abs(low - 0.8) < 1e-9 and abs(high - 3.2) < 1e-9
