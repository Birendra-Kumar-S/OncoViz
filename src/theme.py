"""
Theme constants for OncoViz.

Provides a consistent color palette and Plotly layout defaults
used across all chart and UI components.
"""

# ===================================================================
# Chart colors
# ===================================================================

# Colorblind-safe palette — no pure red or green
CHART_COLORS = ["#0077B6", "#E76F51", "#2A9D8F", "#F4A261", "#6A4C93",
                "#264653", "#B5838D", "#F2CC8F", "#3D405B", "#5FA8D3"]


def get_chart_colors() -> list:
    """Return the chart color palette."""
    return CHART_COLORS


def get_plotly_layout_args() -> dict:
    """Return standard Plotly layout kwargs."""
    return {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#1A1A2E"},
    }
