"""
OncoViz — Main Streamlit Application

Interactive Clinical Trial Data Explorer combining statistical
analysis with automated figure generation.

Run with: streamlit run app.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OncoViz",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Red sliders ---- */
/* Thumb */
[data-testid="stSlider"] [role="slider"] {
    background-color: #D62828 !important;
    border-color: #D62828 !important;
}
/* Filled track (left of thumb) — target all inner divs */
[data-testid="stSlider"] [role="progressbar"] > div:first-child {
    background-color: #D62828 !important;
}
/* Unfilled track (right of thumb) */
[data-testid="stSlider"] [role="progressbar"] {
    background-color: #F0A0A0 !important;
}
/* Tick bar below slider */
[data-testid="stSlider"] [data-testid="stTickBar"] {
    background: rgba(214,40,40,0.3) !important;
}

/* ---- Pill-button tabs ---- */
button[data-baseweb="tab"] {
    background-color: #F0F2F6 !important;
    border: 1.5px solid #DEE2E6 !important;
    border-radius: 8px !important;
    margin-right: 6px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    color: #555 !important;
    transition: all 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    background-color: #E2E6EA !important;
    border-color: #C0C4C8 !important;
    color: #333 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #0077B6 !important;
    border-color: #0077B6 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(0,119,182,0.3) !important;
}
/* Hide the default underline highlight bar */
div[data-baseweb="tab-highlight"] {
    display: none !important;
}
div[data-baseweb="tab-border"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Imports (after set_page_config)
# ---------------------------------------------------------------------------
from src.components.sidebar import render_sidebar
from src.components.stats_panel import render_stats_panel
from src.components.viz_panel import render_viz_panel
from src.utils.helpers import dataset_summary, dataframe_with_csv
from src.data_loader import get_variable_description


# ===================================================================
# Main application
# ===================================================================

def main():
    # ---- Sidebar: dataset & module selection ----
    ctx = render_sidebar()

    df = ctx["df"]
    col_metadata = ctx["col_metadata"]
    dataset_name = ctx["dataset_name"]
    active_module = ctx["active_module"]

    # ---- Header ----
    st.title(":microscope: OncoViz — Interactive Clinical Trial Data Explorer")
    st.caption(
        f"Exploring **{dataset_name}** — "
        f"{len(df)} patients, {len(df.columns)} variables"
    )

    # ---- Data overview section (always visible) ----
    with st.expander("Dataset Overview", expanded=False):
        _render_data_overview(df, col_metadata)

    st.markdown("---")

    # ---- Route to active module ----
    if active_module == "stats":
        render_stats_panel(df, col_metadata, dataset_name)
    elif active_module == "viz":
        render_viz_panel(df, col_metadata)


# ===================================================================
# Data overview
# ===================================================================

def _render_data_overview(df, col_metadata):
    """Render a quick data overview section."""

    tab_summary, tab_raw, tab_desc = st.tabs([
        "Summary Statistics",
        "Raw Data",
        "Data Dictionary",
    ])

    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Numerical", len(col_metadata["numerical"]))
        col4.metric("Missing Cells", int(df.isna().sum().sum()))

        st.markdown("**Numerical columns**")
        num_cols = col_metadata["numerical"]
        if num_cols:
            dataframe_with_csv(
                df[num_cols].describe().round(3),
                key="overview_summary",
                filename="oncoviz_summary_stats",
            )

    with tab_raw:
        dataframe_with_csv(
            df, key="overview_raw", filename="oncoviz_raw_data", height=400,
        )

    with tab_desc:
        summary = dataset_summary(df)
        dataframe_with_csv(
            summary, key="overview_dict", filename="oncoviz_data_dict",
            hide_index=True,
        )


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    main()
