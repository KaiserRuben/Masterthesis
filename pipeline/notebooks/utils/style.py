"""
Monochromatic design system for boundary visualizations.

Color is reserved exclusively for data encoding.
"""

from typing import Any


# Monochromatic theme
THEME = {
    # Base palette (grayscale only)
    "bg": "#fafafa",
    "surface": "#ffffff",
    "text": "#1a1a1a",
    "text_secondary": "#666666",
    "text_muted": "#999999",
    "border": "#e0e0e0",
    "grid": "#eeeeee",

    # Data encoding colors (use sparingly)
    "ade": {
        "low": "#2d9a4d",       # Green - good
        "medium": "#d4a017",    # Amber - caution
        "high": "#d35400",      # Orange - warning
        "critical": "#c0392b", # Red - failure
        "missing": "#cccccc",   # Gray - no data
    },

    # Sequential scale for continuous ADE
    "ade_scale": [
        [0.0, "#2d9a4d"],
        [0.3, "#d4a017"],
        [0.6, "#d35400"],
        [1.0, "#c0392b"],
    ],

    # Categorical scale for semantic keys (muted, distinguishable)
    "categorical": [
        "#5b7fa3",  # Steel blue
        "#7a8b6e",  # Sage
        "#a07a5c",  # Taupe
        "#8b7a9e",  # Dusty purple
        "#6a9a96",  # Teal
        "#9e8a7a",  # Warm gray
    ],

    # Emphasis colors
    "accent": "#1a1a1a",
    "highlight": "#e74c3c",
    "danger_glow": "rgba(199, 57, 43, 0.25)",

    # Typography
    "font_family": "Inter, -apple-system, system-ui, sans-serif",
    "font_mono": "JetBrains Mono, Menlo, monospace",
}


def plotly_layout(
    title: str = "",
    height: int = 500,
    width: int | None = None,
    show_legend: bool = True,
    margin: dict | None = None,
) -> dict[str, Any]:
    """
    Generate consistent Plotly layout configuration.

    Args:
        title: Chart title (centered)
        height: Figure height in pixels
        width: Figure width (None for responsive)
        show_legend: Whether to show legend
        margin: Custom margins dict

    Returns:
        Layout dict for fig.update_layout()
    """
    layout = {
        "paper_bgcolor": THEME["bg"],
        "plot_bgcolor": THEME["bg"],
        "font": {
            "family": THEME["font_family"],
            "color": THEME["text"],
            "size": 12,
        },
        "title": {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": THEME["text"]},
        } if title else None,
        "height": height,
        "showlegend": show_legend,
        "legend": {
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": THEME["border"],
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "margin": margin or {"l": 60, "r": 40, "t": 60, "b": 50},
    }

    if width:
        layout["width"] = width

    return {k: v for k, v in layout.items() if v is not None}


def axis_style(title: str = "", show_grid: bool = True) -> dict[str, Any]:
    """Generate consistent axis styling."""
    return {
        "title": {"text": title, "font": {"size": 12}},
        "gridcolor": THEME["grid"] if show_grid else "rgba(0,0,0,0)",
        "linecolor": THEME["border"],
        "tickfont": {"size": 10, "color": THEME["text_secondary"]},
        "zerolinecolor": THEME["border"],
    }


def scene_style() -> dict[str, Any]:
    """Generate 3D scene styling."""
    return {
        "bgcolor": THEME["bg"],
        "xaxis": {**axis_style("UMAP 1"), "backgroundcolor": THEME["bg"]},
        "yaxis": {**axis_style("UMAP 2"), "backgroundcolor": THEME["bg"]},
        "zaxis": {**axis_style("UMAP 3"), "backgroundcolor": THEME["bg"]},
    }


def format_pct(value: float, decimals: int = 1) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_count(value: int) -> str:
    """Format integer with thousand separators."""
    return f"{value:,}"
