"""
Shared utility functions for OncoViz.

Provides formatting helpers, column type detection, and export utilities
used by both the statistics and visualization modules.
"""

import io
import numpy as np
import pandas as pd


# ===================================================================
# Formatting
# ===================================================================

def format_p_value(p: float) -> str:
    """
    Format a p-value for display.

    Examples
    --------
    >>> format_p_value(0.0001)
    'p < 0.001'
    >>> format_p_value(0.034)
    'p = 0.034'
    """
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.1:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def format_stat(value: float, name: str = "stat") -> str:
    """Format a test statistic for display."""
    return f"{name} = {value:.4f}"


def significance_badge(p: float, alpha: float = 0.05) -> str:
    """
    Return a text badge for significance level.

    Returns one of: '✱✱✱ (p<0.001)', '✱✱ (p<0.01)',
    '✱ (p<0.05)', or 'n.s.'
    """
    if p < 0.001:
        return "*** (p < 0.001)"
    elif p < 0.01:
        return "** (p < 0.01)"
    elif p < alpha:
        return "* (p < 0.05)"
    else:
        return "n.s."


def interpret_effect_size_text(effect_type: str, value: float) -> str:
    """
    Return a descriptive interpretation of an effect size.

    Parameters
    ----------
    effect_type : str — "cohens_d", "eta_squared", "cramers_v", "r"
    value : float
    """
    if np.isnan(value):
        return "Effect size could not be calculated."

    v = abs(value)

    if effect_type == "cohens_d":
        if v < 0.2:
            mag = "negligible"
        elif v < 0.5:
            mag = "small"
        elif v < 0.8:
            mag = "medium"
        else:
            mag = "large"
        return f"Cohen's d = {value:.3f} ({mag} effect)"

    elif effect_type == "eta_squared":
        if v < 0.01:
            mag = "negligible"
        elif v < 0.06:
            mag = "small"
        elif v < 0.14:
            mag = "medium"
        else:
            mag = "large"
        return f"Eta-squared = {value:.3f} ({mag} effect)"

    elif effect_type == "cramers_v":
        if v < 0.1:
            mag = "negligible"
        elif v < 0.3:
            mag = "small"
        elif v < 0.5:
            mag = "medium"
        else:
            mag = "large"
        return f"Cramer's V = {value:.3f} ({mag} association)"

    elif effect_type == "r":
        if v < 0.1:
            mag = "negligible"
        elif v < 0.3:
            mag = "weak"
        elif v < 0.5:
            mag = "moderate"
        elif v < 0.7:
            mag = "strong"
        else:
            mag = "very strong"
        direction = "positive" if value > 0 else "negative"
        return f"r = {value:.3f} ({mag} {direction} correlation)"

    return f"Effect size = {value:.3f}"


# ===================================================================
# Column classification
# ===================================================================

def classify_column(series: pd.Series) -> str:
    """
    Classify a pandas Series as 'numerical', 'categorical', or 'binary'.

    Rules
    -----
    - Binary: exactly 2 unique non-null values
    - Numerical: numeric dtype with > 10 unique values
    - Categorical: everything else (including low-cardinality numerics)
    """
    n_unique = series.dropna().nunique()

    if n_unique <= 2 and n_unique > 0:
        return "binary"
    elif pd.api.types.is_numeric_dtype(series) and n_unique > 10:
        return "numerical"
    else:
        return "categorical"


# ===================================================================
# Data summary helpers
# ===================================================================

def describe_column(series: pd.Series) -> dict:
    """
    Generate a summary dict for a single column, adapting to its type.
    """
    col_type = classify_column(series)
    n_total = len(series)
    n_missing = int(series.isna().sum())

    base = {
        "type": col_type,
        "n_total": n_total,
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_total * 100, 1) if n_total > 0 else 0,
    }

    if col_type == "numerical":
        clean = series.dropna()
        base.update({
            "mean": round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std": round(float(clean.std()), 4),
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
            "q1": round(float(clean.quantile(0.25)), 4),
            "q3": round(float(clean.quantile(0.75)), 4),
            "skew": round(float(clean.skew()), 4),
        })
    else:
        counts = series.value_counts(dropna=False)
        base.update({
            "n_unique": int(series.dropna().nunique()),
            "top_values": counts.head(10).to_dict(),
            "mode": str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
        })

    return base


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame with one row per column showing type,
    non-null count, unique values, and sample values.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "Column": col,
            "Type": classify_column(s),
            "Non-Null": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Unique": int(s.nunique()),
            "Sample Values": ", ".join(str(v) for v in s.dropna().unique()[:4]),
        })
    return pd.DataFrame(rows)


# ===================================================================
# Export helpers
# ===================================================================

def fig_to_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """
    Convert a Plotly or Matplotlib figure to image bytes for download.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
    fmt : str — "png", "svg", or "pdf"
    dpi : int — resolution (matplotlib only)

    Returns
    -------
    bytes
    """
    # Plotly
    if hasattr(fig, "to_image"):
        return fig.to_image(format=fmt, scale=2)

    # Matplotlib
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ===================================================================
# Download helpers for Streamlit UI
# ===================================================================

def _kaleido_available() -> bool:
    """Check once whether kaleido is importable (cached in module global)."""
    try:
        import kaleido  # noqa: F401
        return True
    except ImportError:
        return False


def figure_download_ui(fig, key: str, filename: str = "oncoviz_figure"):
    """
    Render a figure download section below a Plotly chart.
    Offers PNG, JPEG, SVG via kaleido.  If kaleido is missing the
    dropdown still shows but exports the interactive HTML instead,
    with a small note telling the user how to enable image export.
    """
    import streamlit as _st

    has_kaleido = _kaleido_available()

    _c1, _c2, _c3 = _st.columns([6, 2, 2])
    with _c2:
        fmt = _st.selectbox(
            "Format",
            options=["png", "jpeg", "svg"],
            key=f"figfmt_{key}",
            label_visibility="collapsed",
        )
    with _c3:
        if has_kaleido:
            buf = io.BytesIO()
            fig.write_image(buf, format=fmt, scale=3, width=1200, height=700)
            buf.seek(0)
            mime_map = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "svg": "image/svg+xml",
            }
            _st.download_button(
                "Download Figure",
                data=buf,
                file_name=f"{filename}.{fmt}",
                mime=mime_map[fmt],
                key=f"figdl_{key}",
            )
        else:
            # kaleido not installed — export as interactive HTML
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode()
            _st.download_button(
                "Download Figure",
                data=html_bytes,
                file_name=f"{filename}.html",
                mime="text/html",
                key=f"figdl_{key}",
            )
    if not has_kaleido:
        _st.caption(
            "Install **kaleido** for PNG/JPEG/SVG export: "
            "`pip install -U kaleido`"
        )


def dataframe_with_csv(
    df_display: pd.DataFrame,
    key: str,
    filename: str = "data",
    hide_index: bool = False,
    height: int = None,
):
    """
    Display a DataFrame with a CSV download button underneath.
    Drop-in replacement for st.dataframe() with export.
    """
    import streamlit as _st

    kwargs = {"use_container_width": True}
    if hide_index:
        kwargs["hide_index"] = True
    if height:
        kwargs["height"] = height

    _st.dataframe(df_display, **kwargs)

    csv_data = df_display.to_csv()
    _st.download_button(
        "Download CSV",
        data=csv_data,
        file_name=f"{filename}.csv",
        mime="text/csv",
        key=f"csv_{key}",
    )
