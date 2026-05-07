"""
Sidebar component for OncoViz.

Provides dataset selection, data overview, and navigation
between the Statistical Analysis and Figure Generator modules.
"""

import streamlit as st
import pandas as pd

from src.data_loader import (
    get_dataset_names,
    get_dataset_info,
    load_dataset,
    get_column_metadata,
    get_variable_description,
)
from src.utils.helpers import dataset_summary


# ===================================================================
# Main sidebar renderer
# ===================================================================

def render_sidebar() -> dict:
    """
    Render the sidebar and return a dict with the user's selections:
        {
            "dataset_name": str,
            "df": pd.DataFrame,
            "col_metadata": dict,
            "active_module": str,   # "stats" | "viz"
        }
    """

    st.sidebar.markdown("## OncoViz")
    st.sidebar.caption("Interactive Clinical Trial Data Explorer")
    st.sidebar.markdown("---")

    # ---- Dataset selection ----
    st.sidebar.markdown("### Dataset")
    dataset_names = get_dataset_names()
    dataset_name = st.sidebar.selectbox(
        "Choose a dataset",
        options=dataset_names,
        key="sidebar_dataset",
    )

    # Load data
    df = load_dataset(dataset_name)
    info = get_dataset_info(dataset_name)
    col_metadata = get_column_metadata(df)

    # Dataset info
    st.sidebar.caption(info["description"])
    st.sidebar.markdown(
        f"**Rows:** {len(df)} &nbsp;|&nbsp; **Columns:** {len(df.columns)}"
    )
    st.sidebar.markdown(f"**Source:** {info['source']}")

    st.sidebar.markdown("---")

    # ---- Module selection ----
    st.sidebar.markdown("### Module")
    active_module = st.sidebar.radio(
        "Select module",
        options=["stats", "viz"],
        format_func=lambda x: {
            "stats": "Statistical Analysis",
            "viz": "Automated Figure Maker",
        }[x],
        key="sidebar_module",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # ---- Quick data preview in sidebar ----
    with st.sidebar.expander("Data Preview"):
        st.dataframe(df.head(10), use_container_width=True, height=300)

    with st.sidebar.expander("Column Types"):
        for category, label in [
            ("numerical", "Numerical"),
            ("categorical", "Categorical"),
            ("binary", "Binary"),
        ]:
            cols = col_metadata[category]
            if cols:
                st.markdown(f"**{label}** ({len(cols)})")
                for c in cols:
                    st.caption(f"  {c} — {get_variable_description(c)}")

    return {
        "dataset_name": dataset_name,
        "df": df,
        "col_metadata": col_metadata,
        "active_module": active_module,
    }
