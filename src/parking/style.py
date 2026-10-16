"""Colours shared by the maps and the demo: a dark slate background and one blue accent."""
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

BACKGROUND = "#0e1218"
PANEL = "#151b23"
GRID = "#232b36"
TEXT = "#e8ecf1"
MUTED = "#8b96a5"
ACCENT = "#5b9dff"
OUTLINE = "#f5f7fa"

# Low rates fade into the background, high rates reach the accent and then a pale blue.
RATE_MAP = LinearSegmentedColormap.from_list("rate", ["#1b2a44", "#2f5fb3", ACCENT, "#cfe4ff"])


def padded_limits(values, padding: float = 0.02) -> tuple[float, float]:
    """Axis limits covering all values with a small margin."""
    low, high = float(values.min()), float(values.max())
    margin = (high - low) * padding
    return low - margin, high + margin


def style_axes(ax: Axes) -> None:
    """Dark panel, thin grid, muted labels."""
    ax.set_facecolor(PANEL)
    ax.grid(color=GRID, linewidth=0.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
