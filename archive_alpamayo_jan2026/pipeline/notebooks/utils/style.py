"""
Monochromatic design system for boundary visualizations.

Color is reserved exclusively for data encoding.
"""

from typing import Any


# Design system: monochrome chrome, purposeful data color
THEME = {
    # Chrome palette (grayscale) - for UI, not data
    "bg": "#fafafa",
    "surface": "#ffffff",
    "text": "#1a1a1a",
    "text_secondary": "#666666",
    "text_muted": "#999999",
    "border": "#e0e0e0",
    "grid": "#eeeeee",
    "point_inactive": "#d0d0d0",  # Background points when not the focus

    # ==========================================================================
    # DATA ENCODING COLORS
    # In 3D visualizations, color is essential for perception - not decorative.
    # Without color, depth ambiguity makes clusters indistinguishable.
    # ==========================================================================

    # ADE classes: semantic traffic-light progression
    # Green = safe, Red = failure. Universally understood.
    "ade": {
        "low": "#3a9e5c",       # Forest green - good prediction
        "medium": "#e6a132",    # Amber - acceptable
        "high": "#d66b2b",      # Burnt orange - poor
        "critical": "#bf3636",  # Crimson - failure
        "missing": "#c8c8c8",   # Neutral gray - no data
    },

    # Sequential scale for continuous ADE (interpolated)
    "ade_scale": [
        [0.0, "#3a9e5c"],
        [0.33, "#e6a132"],
        [0.66, "#d66b2b"],
        [1.0, "#bf3636"],
    ],

    # Categorical palette for semantic classes
    # Desaturated, high-contrast, colorblind-accessible
    # Each color must be distinguishable in 3D point clouds
    "categorical": [
        "#4878a8",  # Steel blue
        "#6a9a58",  # Moss green
        "#c87040",  # Terracotta
        "#7868a8",  # Muted purple
        "#48989c",  # Teal
        "#b87878",  # Dusty rose
        "#8a8a5a",  # Olive
        "#a86088",  # Mauve
    ],

    # Binary encoding (anchor vs propagated)
    "binary": {
        True: "#2c5aa0",   # Anchor: distinct blue
        False: "#b8b8b8",  # Propagated: recedes
    },

    # Emphasis and alerts
    "accent": "#1a1a1a",
    "highlight": "#c94444",
    "danger_glow": "rgba(191, 54, 54, 0.3)",

    # Typography
    "font_family": "Inter, -apple-system, system-ui, sans-serif",
    "font_mono": "JetBrains Mono, Menlo, monospace",

    # Diverging colorscale (for asymmetry, difference plots)
    "diverging": {
        "negative": "#4a7c59",  # Muted sage green
        "neutral": "#fafafa",   # Background
        "positive": "#8b4a4a",  # Muted burgundy
    },

    # Grayscale sequence (for multi-line plots, ordered data)
    "grays": ["#1a1a1a", "#4a4a4a", "#6a6a6a", "#8a8a8a", "#aaaaaa"],

    # Grayscale colorscale (for continuous heatmaps)
    "grayscale": [
        [0.0, "#f0f0f0"],
        [1.0, "#1a1a1a"],
    ],
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
    """
    Generate 3D scene styling.

    Chrome is monochrome (grid, axes, background) so colored data points
    stand out clearly against neutral background.
    """
    axis_common = {
        "backgroundcolor": THEME["bg"],
        "gridcolor": THEME["grid"],
        "linecolor": THEME["border"],
        "tickfont": {"size": 9, "color": THEME["text_muted"]},
        "title": {"font": {"size": 11, "color": THEME["text_secondary"]}},
    }
    return {
        "bgcolor": THEME["bg"],
        "xaxis": {**axis_common, "title": {"text": "UMAP 1", **axis_common["title"]}},
        "yaxis": {**axis_common, "title": {"text": "UMAP 2", **axis_common["title"]}},
        "zaxis": {**axis_common, "title": {"text": "UMAP 3", **axis_common["title"]}},
    }


def format_pct(value: float, decimals: int = 1) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_count(value: int) -> str:
    """Format integer with thousand separators."""
    return f"{value:,}"
