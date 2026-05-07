"""
Automated Figure Maker panel for OncoViz.

Provides an interactive interface for creating publication-quality figures
from clinical trial data. Supports 4 plot types with smart variable
suggestions, customization options, and figure export.
"""

import io

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import get_variable_description
from src.theme import get_chart_colors, get_plotly_layout_args
from src.utils.helpers import figure_download_ui, dataframe_with_csv


# ===================================================================
# Color palette presets — colorblind-safe (no pure red / green)
# ===================================================================

PALETTE_OPTIONS = {
    # --- Single colors (for plots without group split) ---
    "Steel Blue": ["#0077B6"],
    "Teal": ["#2A9D8F"],
    "Orange": ["#F4A261"],
    "Purple": ["#6A4C93"],
    "Dark Navy": ["#3D405B"],
    # --- Multi-color palettes (curated: no red, no green) ---
    "OncoViz Default": None,  # uses theme colors (already safe)
    "Viridis Safe": [
        "#440154", "#482878", "#3e4989", "#31688e",
        "#26828e", "#1f9e89", "#fde725",
    ],
    "Plasma": [
        "#0d0887", "#46039f", "#7201a8", "#9c179e",
        "#bd3786", "#d8576b", "#ed7953", "#fb9f3a",
        "#fdca26", "#f0f921",
    ],
    "Blues": px.colors.sequential.Blues,
    "Ocean": [
        "#003f5c", "#2f4b7c", "#665191", "#a05195",
        "#d45087", "#f95d6a", "#ff7c43", "#ffa600",
    ],
    "Pastel Safe": [
        "rgb(102, 197, 204)", "rgb(246, 207, 113)",
        "rgb(248, 156, 116)", "rgb(220, 176, 242)",
        "rgb(158, 185, 243)", "rgb(254, 136, 177)",
        "rgb(180, 151, 231)", "rgb(179, 179, 179)",
    ],
    "Bold Safe": [
        "rgb(127, 60, 141)", "rgb(57, 105, 172)",
        "rgb(242, 183, 1)", "rgb(230, 131, 16)",
        "rgb(0, 134, 149)", "rgb(207, 28, 144)",
        "rgb(165, 170, 153)",
    ],
}


# ===================================================================
# Main entry point
# ===================================================================

def render_viz_panel(df: pd.DataFrame, col_metadata: dict):
    """Render the automated figure generation interface."""

    st.header("Automated Figure Maker")
    st.markdown(
        "Create publication-quality figures from your clinical trial data. "
        "Select a plot type, map variables, customize the look, and export."
    )

    plot_types = [
        "Distribution",
        "Categorical Plot",
        "Heatmap",
    ]

    tabs = st.tabs(plot_types)

    with tabs[0]:
        _render_distribution(df, col_metadata)
    with tabs[1]:
        _render_categorical_plot(df, col_metadata)
    with tabs[2]:
        _render_heatmap(df, col_metadata)


# ===================================================================
# Shared helpers
# ===================================================================

def _get_color_sequence(palette_name: str) -> list:
    """Return a color list for the chosen palette."""
    if palette_name == "OncoViz Default" or palette_name not in PALETTE_OPTIONS:
        return get_chart_colors()
    return PALETTE_OPTIONS[palette_name]


def _customization_sidebar(key_prefix: str, show_palette: bool = True):
    """
    Render common customization controls inside an expander.
    Returns dict with user choices.
    """
    with st.expander("Customize Appearance", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input(
                "Chart title",
                value="",
                placeholder="Auto-generated if blank",
                key=f"{key_prefix}_title",
            )
        with c2:
            if show_palette:
                palette = st.selectbox(
                    "Color palette",
                    options=list(PALETTE_OPTIONS.keys()),
                    index=5,  # default to "OncoViz Default"
                    key=f"{key_prefix}_palette",
                )
            else:
                palette = "OncoViz Default"

        c3, c4 = st.columns(2)
        with c3:
            height = st.slider(
                "Chart height (px)",
                min_value=300, max_value=900, value=500, step=50,
                key=f"{key_prefix}_height",
            )
        with c4:
            opacity = st.slider(
                "Opacity",
                min_value=0.1, max_value=1.0, value=0.75, step=0.05,
                key=f"{key_prefix}_opacity",
            )

    return {
        "title": title,
        "palette": palette,
        "height": height,
        "opacity": opacity,
    }


# ===================================================================
# 1. Distribution Plot (Histogram + KDE)
# ===================================================================

def _render_distribution(df: pd.DataFrame, meta: dict):
    """Interactive distribution plot with histogram and optional KDE overlay."""

    st.subheader("Distribution Plot")
    st.caption(
        "Explore the distribution of a numerical variable. "
        "Optionally split by a categorical variable to compare groups."
    )

    # Variable selectors
    col1, col2 = st.columns(2)
    with col1:
        num_col = st.selectbox(
            "Numerical variable",
            options=meta["numerical"],
            key="dist_num",
            help="Choose the continuous variable to plot.",
        )
    with col2:
        group_options = ["None"] + meta["categorical"] + meta["binary"]
        group_col = st.selectbox(
            "Split by (optional)",
            options=group_options,
            key="dist_group",
            help="Overlay distributions for each group.",
        )

    if not num_col:
        return

    group_var = None if group_col == "None" else group_col

    # Display options
    c1, c2, c3 = st.columns(3)
    with c1:
        show_hist = st.checkbox("Histogram", value=True, key="dist_hist")
    with c2:
        show_kde = st.checkbox("KDE curve", value=True, key="dist_kde")
    with c3:
        show_rug = st.checkbox("Rug plot", value=False, key="dist_rug")

    c4, c5 = st.columns(2)
    with c4:
        n_bins = st.slider(
            "Number of bins", min_value=5, max_value=100, value=30,
            key="dist_bins",
        )
    with c5:
        if group_var:
            bar_mode = st.radio(
                "Split display mode",
                options=["Side-by-Side", "Overlay"],
                key="dist_barmode",
                horizontal=True,
                help="Side-by-Side: bars next to each other (clearer). "
                     "Overlay: bars on top of each other (transparent).",
            )
        else:
            bar_mode = "Side-by-Side"

    # Customization
    opts = _customization_sidebar("dist")
    colors = _get_color_sequence(opts["palette"])
    layout_args = get_plotly_layout_args()

    plot_df = df[[num_col] + ([group_var] if group_var else [])].dropna()

    # Determine barmode
    plotly_barmode = "group" if bar_mode == "Side-by-Side" else "overlay"
    # Force lower opacity when overlay mode with split
    eff_opacity = min(opts["opacity"], 0.55) if (
        plotly_barmode == "overlay" and group_var
    ) else opts["opacity"]

    if not show_hist and not show_kde:
        st.warning("Select at least Histogram or KDE curve.")
        return

    # Build the figure
    if show_hist and show_kde:
        fig = px.histogram(
            plot_df, x=num_col, color=group_var,
            nbins=n_bins, marginal="violin",
            histnorm="probability density",
            barmode=plotly_barmode,
            color_discrete_sequence=colors,
            opacity=eff_opacity,
        )
    elif show_hist:
        fig = px.histogram(
            plot_df, x=num_col, color=group_var,
            nbins=n_bins,
            barmode=plotly_barmode,
            color_discrete_sequence=colors,
            opacity=eff_opacity,
        )
    else:
        # KDE only — use violin as a proxy
        fig = px.violin(
            plot_df, x=num_col, color=group_var,
            color_discrete_sequence=colors,
        )

    # Add rug plot markers if requested
    if show_rug and show_hist:
        if group_var:
            for i, grp in enumerate(plot_df[group_var].unique()):
                vals = plot_df.loc[plot_df[group_var] == grp, num_col]
                fig.add_trace(go.Scatter(
                    x=vals, y=[-0.01] * len(vals),
                    mode="markers",
                    marker=dict(symbol="line-ns-open", size=8,
                                color=colors[i % len(colors)]),
                    showlegend=False,
                    hoverinfo="x",
                ))
        else:
            fig.add_trace(go.Scatter(
                x=plot_df[num_col], y=[-0.01] * len(plot_df),
                mode="markers",
                marker=dict(symbol="line-ns-open", size=8, color=colors[0]),
                showlegend=False,
                hoverinfo="x",
            ))

    title = opts["title"] or f"Distribution of {num_col}" + (
        f" by {group_var}" if group_var else ""
    )

    fig.update_layout(
        title=title,
        xaxis_title=num_col,
        yaxis_title="Density" if show_kde else "Count",
        height=opts["height"],
        **layout_args,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Download figure
    figure_download_ui(fig, key="dist", filename=f"distribution_{num_col}")

    # Summary stats below the plot
    with st.expander("Summary Statistics"):
        if group_var:
            summary = plot_df.groupby(group_var)[num_col].describe().round(3)
        else:
            summary = plot_df[num_col].describe().round(3).to_frame().T
        dataframe_with_csv(summary, key="dist_summary", filename=f"dist_summary_{num_col}")


# ===================================================================
# 2. Categorical Plot (Counts / Proportions)
# ===================================================================

def _render_categorical_plot(df: pd.DataFrame, meta: dict):
    """
    Unified categorical bar / proportion plot.
    Supports counts and percentages, side-by-side / stacked / 100% stacked,
    with optional grouping by a second variable.
    """

    st.subheader("Categorical Plot")
    st.caption(
        "Visualize counts or proportions of categorical variables. "
        "Optionally group by a second variable to compare composition."
    )

    cat_options = meta["categorical"] + meta["binary"]

    col1, col2 = st.columns(2)
    with col1:
        primary_col = st.selectbox(
            "Primary variable (x-axis)",
            options=cat_options,
            key="cat_primary",
        )
    with col2:
        secondary_options = ["None"] + [c for c in cat_options if c != primary_col]
        group_col = st.selectbox(
            "Group by (color)",
            options=secondary_options,
            key="cat_group",
        )

    if not primary_col:
        return

    has_group = group_col != "None"

    # --- Layout options ---
    if has_group:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bar_mode_label = st.radio(
                "Bar mode",
                options=["Side by Side", "Stacked"],
                key="cat_barmode",
                horizontal=True,
            )
        with c2:
            display_as = st.radio(
                "Display as",
                options=["Counts", "Percentage (%)"],
                key="cat_display",
                horizontal=True,
            )
        with c3:
            orientation = st.radio(
                "Orientation",
                options=["Vertical", "Horizontal"],
                key="cat_orient",
                horizontal=True,
            )
        with c4:
            show_values = st.checkbox("Show values", value=True, key="cat_values")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            display_as = st.radio(
                "Display as",
                options=["Counts", "Percentage (%)"],
                key="cat_display",
                horizontal=True,
            )
        with c2:
            orientation = st.radio(
                "Orientation",
                options=["Vertical", "Horizontal"],
                key="cat_orient",
                horizontal=True,
            )
        with c3:
            show_values = st.checkbox("Show values", value=True, key="cat_values")
        bar_mode_label = "Side by Side"  # not applicable for single variable

    # Customization
    opts = _customization_sidebar("cat")
    colors = _get_color_sequence(opts["palette"])
    layout_args = get_plotly_layout_args()

    is_horizontal = orientation == "Horizontal"
    is_pct = display_as == "Percentage (%)"

    # Map bar mode label to plotly barmode
    barmode = "group" if bar_mode_label == "Side by Side" else "stack"

    # --- Build the figure ---
    if has_group:
        cols_needed = [primary_col, group_col]
        plot_df = df[cols_needed].dropna()
        ct = pd.crosstab(plot_df[primary_col], plot_df[group_col])

        if is_pct:
            plot_data = ct.div(ct.sum(axis=1), axis=0) * 100
            value_label = "Percentage (%)"
            text_fmt = ".1f"
        else:
            plot_data = ct
            value_label = "Count"
            text_fmt = True

        fig = px.bar(
            plot_data,
            barmode=barmode,
            orientation="h" if is_horizontal else "v",
            color_discrete_sequence=colors,
            text_auto=text_fmt if show_values else False,
        )
        fig.update_layout(
            xaxis_title=value_label if is_horizontal else primary_col,
            yaxis_title=primary_col if is_horizontal else value_label,
            legend_title=group_col,
        )

        auto_title = f"{primary_col} by {group_col}"
    else:
        plot_df = df[[primary_col]].dropna()
        counts = plot_df[primary_col].value_counts().sort_index()

        if is_pct:
            values = (counts / counts.sum() * 100).round(1)
            value_label = "Percentage (%)"
            text_fmt = ".1f"
        else:
            values = counts
            value_label = "Count"
            text_fmt = True

        if is_horizontal:
            fig = px.bar(
                x=values.values, y=values.index,
                orientation="h",
                color_discrete_sequence=colors,
                text=values.values if show_values else None,
            )
            fig.update_layout(xaxis_title=value_label, yaxis_title=primary_col)
        else:
            fig = px.bar(
                x=values.index, y=values.values,
                color_discrete_sequence=colors,
                text=values.values if show_values else None,
            )
            fig.update_layout(xaxis_title=primary_col, yaxis_title=value_label)

        auto_title = f"{value_label} of {primary_col}"

    title = opts["title"] or auto_title
    fig.update_layout(title=title, height=opts["height"], **layout_args)
    fig.update_traces(opacity=opts["opacity"])

    st.plotly_chart(fig, use_container_width=True)

    # Download figure
    figure_download_ui(fig, key="cat_plot", filename=f"categorical_{primary_col}")

    # Frequency / contingency table
    with st.expander("Data Table"):
        if has_group:
            ct_display = pd.crosstab(
                plot_df[primary_col], plot_df[group_col], margins=True,
            )
            dataframe_with_csv(
                ct_display, key="cat_ct",
                filename=f"contingency_{primary_col}_{group_col}",
            )
            if is_pct:
                st.markdown("**Proportions (%)**")
                ct_pct = pd.crosstab(
                    plot_df[primary_col], plot_df[group_col], normalize="index",
                ).round(3) * 100
                dataframe_with_csv(
                    ct_pct, key="cat_pct",
                    filename=f"proportions_{primary_col}_{group_col}",
                )
        else:
            freq = plot_df[primary_col].value_counts().reset_index()
            freq.columns = [primary_col, "Count"]
            freq["Percentage"] = (freq["Count"] / freq["Count"].sum() * 100).round(1)
            dataframe_with_csv(
                freq, key="cat_freq",
                filename=f"freq_{primary_col}",
                hide_index=True,
            )


# ===================================================================
# 3. Heatmap (Patient-Feature)
# ===================================================================

def _render_heatmap(df: pd.DataFrame, meta: dict):
    """
    Heatmap of standardized clinical features across patients.
    Each row is a patient, each column is a z-scored numerical variable.
    """

    st.subheader("Clinical Feature Heatmap")
    st.caption(
        "Visualize standardized clinical measurements across patients. "
        "Each cell shows the z-score (standard deviations from the mean). "
        "Helps identify patient subgroups and variable patterns."
    )

    num_cols = meta["numerical"]
    if len(num_cols) < 2:
        st.warning("Need at least 2 numerical variables for a heatmap.")
        return

    # Variable selection
    selected_vars = st.multiselect(
        "Select clinical variables",
        options=num_cols,
        default=num_cols[:min(6, len(num_cols))],
        key="hm_vars",
    )

    if len(selected_vars) < 2:
        st.info("Select at least 2 variables.")
        return

    # Sorting and grouping
    c1, c2 = st.columns(2)
    with c1:
        sort_options = ["None"] + meta["categorical"] + meta["binary"]
        sort_by = st.selectbox(
            "Group patients by",
            options=sort_options,
            key="hm_sort",
            help="Sort patients by a categorical variable to reveal group patterns.",
        )
    with c2:
        color_scale = st.selectbox(
            "Color scale",
            options=["PuOr", "BrBG", "Viridis", "Plasma", "Inferno", "Cividis"],
            key="hm_cscale",
        )

    # Customization
    opts = _customization_sidebar("hm", show_palette=False)
    layout_args = get_plotly_layout_args()

    # Prepare data — include all patients (no slider cap)
    sort_var = None if sort_by == "None" else sort_by
    cols_needed = selected_vars + ([sort_var] if sort_var else [])
    plot_df = df[cols_needed].dropna()

    # Sort by group variable if selected
    if sort_var:
        plot_df = plot_df.sort_values(sort_var)

    # Z-score standardize
    z_df = plot_df[selected_vars].apply(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0,
        axis=0,
    )

    # Build y-labels with group info
    if sort_var:
        y_labels = [
            f"P{i+1} ({plot_df[sort_var].iloc[i]})"
            for i in range(len(plot_df))
        ]
    else:
        y_labels = [f"Patient {i+1}" for i in range(len(plot_df))]

    n_cols = len(selected_vars)

    fig = px.imshow(
        z_df.values,
        x=[get_variable_description(c).split("(")[0].strip()
           if len(get_variable_description(c)) > 20 else c
           for c in selected_vars],
        y=y_labels,
        color_continuous_scale=color_scale,
        zmin=-3, zmax=3,
        aspect="auto",
    )

    title = opts["title"] or "Clinical Feature Heatmap (Z-scores)"

    # Width scales with number of variables:
    # ~150px per column + margin for y-axis label + colorbar
    fig_width = n_cols * 150 + 200
    fig_width = max(450, min(fig_width, 1400))

    fig.update_layout(
        title=title,
        xaxis_title="Clinical Variable",
        yaxis_title="Patient",
        width=fig_width,
        height=opts["height"],
        coloraxis_colorbar_title="Z-score",
        yaxis=dict(showticklabels=False),
        **layout_args,
    )

    st.plotly_chart(fig, use_container_width=False)

    # Download figure
    figure_download_ui(fig, key="hm", filename="heatmap")

    # Legend
    st.caption(
        "**Reading the heatmap:** Purple/cool = below average, "
        "Orange/warm = above average. Values are z-scored "
        "(0 = mean, ±1 = one standard deviation)."
    )


# ===================================================================
# (Proportion Plot merged into Categorical Plot above)
# ===================================================================

