"""
Statistical Analysis panel for OncoViz.

Provides an interactive interface for selecting variables, choosing
statistical tests (with smart auto-suggestions), displaying results
with p-values, effect sizes, and plain-English interpretations.
"""

import math

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.statistics import (
    run_two_sample_test,
    run_multi_group_test,
    run_chi_square,
    run_correlation,
    run_survival_analysis,
    run_posthoc_pairwise,
    suggest_test,
    suggest_posthoc,
    suggest_correlation_method,
    TestResult,
)
from src.utils.helpers import (
    format_p_value,
    significance_badge,
    interpret_effect_size_text,
    describe_column,
    figure_download_ui,
    dataframe_with_csv,
)
from src.data_loader import get_variable_description, get_dataset_info
from src.theme import get_chart_colors, get_plotly_layout_args


# ===================================================================
# Main entry point
# ===================================================================

def render_stats_panel(df: pd.DataFrame, col_metadata: dict, dataset_name: str):
    """Render the full statistical analysis interface."""

    st.header("Statistical Analysis")
    st.markdown(
        "Select variables and run statistical tests. "
        "OncoViz will recommend the most appropriate test based on "
        "your variable types, or you can override manually."
    )

    # Analysis mode tabs
    tab_compare, tab_correlate, tab_survival = st.tabs([
        "Group Comparison",
        "Correlation",
        "Survival Analysis",
    ])

    with tab_compare:
        _render_comparison_tab(df, col_metadata)

    with tab_correlate:
        _render_correlation_tab(df, col_metadata)

    with tab_survival:
        _render_survival_tab(df, col_metadata, dataset_name)


# ===================================================================
# Group comparison tab
# ===================================================================

def _render_comparison_tab(df: pd.DataFrame, meta: dict):
    """Compare a continuous variable across groups (t-test / ANOVA etc.)."""

    st.subheader("Compare Groups")

    col1, col2 = st.columns(2)

    # Group variable selector (categorical + binary)
    group_options = meta["categorical"] + meta["binary"]
    with col1:
        group_col = st.selectbox(
            "Grouping variable",
            options=group_options,
            help="Categorical or binary variable that defines the groups.",
            key="compare_group",
        )

    # Value variable selector
    with col2:
        value_col = st.selectbox(
            "Measurement variable",
            options=meta["numerical"],
            help="Continuous variable to compare across groups.",
            key="compare_value",
        )

    if not group_col or not value_col:
        st.info("Select both a grouping variable and a measurement variable.")
        return

    # Show variable descriptions
    with st.expander("Variable info"):
        st.markdown(f"**{group_col}**: {get_variable_description(group_col)}")
        st.markdown(f"**{value_col}**: {get_variable_description(value_col)}")

    # Determine number of groups
    n_groups = df[group_col].dropna().nunique()
    group_type = meta["types"].get(group_col, "categorical")

    # Smart suggestion
    suggestion = suggest_test(group_type, "numerical", n_groups)
    st.info(f"**Recommended:** {suggestion['primary']}  \n{suggestion['explanation']}")

    # Test override
    if n_groups == 2:
        test_options = ["auto", "t_test", "mann_whitney"]
        test_labels = {
            "auto": "Auto (checks normality)",
            "t_test": "Independent t-test",
            "mann_whitney": "Mann-Whitney U",
        }
    else:
        test_options = ["auto", "anova", "kruskal"]
        test_labels = {
            "auto": "Auto (checks normality)",
            "anova": "One-way ANOVA",
            "kruskal": "Kruskal-Wallis",
        }

    selected_test = st.selectbox(
        "Test method",
        options=test_options,
        format_func=lambda x: test_labels[x],
        key="compare_test_method",
    )

    # Run test — store result in session_state so it persists across reruns
    if st.button("Run Test", key="btn_compare", type="primary"):
        with st.spinner("Running statistical test..."):
            try:
                if n_groups == 2:
                    result = run_two_sample_test(df, group_col, value_col, test=selected_test)
                else:
                    result = run_multi_group_test(df, group_col, value_col, test=selected_test)

                # Store in session state
                st.session_state["compare_result"] = result
                st.session_state["compare_group_col"] = group_col
                st.session_state["compare_value_col"] = value_col
                st.session_state["compare_n_groups"] = n_groups

            except Exception as e:
                st.error(f"Error: {e}")
                return

    # Display stored results (persists across reruns)
    if "compare_result" in st.session_state:
        result = st.session_state["compare_result"]
        stored_group = st.session_state.get("compare_group_col")
        stored_value = st.session_state.get("compare_value_col")
        stored_n_groups = st.session_state.get("compare_n_groups", 0)

        # Only show if the current selections match the stored result
        if stored_group == group_col and stored_value == value_col:
            _display_test_result(result)
            _display_comparison_plot(df, group_col, value_col)

            # Post-hoc for multi-group tests
            if stored_n_groups > 2:
                _display_posthoc_section(
                    df, group_col, value_col,
                    omnibus_significant=result.significant,
                    omnibus_test_name=result.test_name,
                )


# ===================================================================
# Post-hoc pairwise comparison section
# ===================================================================

def _display_posthoc_section(
    df, group_col, value_col,
    omnibus_significant: bool,
    omnibus_test_name: str,
):
    """
    Show post-hoc pairwise comparisons with multiple testing correction.
    Persists results in session_state so changing the correction dropdown
    doesn't require re-running the omnibus test.
    """
    st.markdown("---")
    st.subheader("Post-hoc Pairwise Comparisons")

    if omnibus_significant:
        st.caption(
            "The omnibus test was significant — pairwise comparisons "
            "identify which specific groups differ. P-values are adjusted "
            "to control for multiple testing."
        )
    else:
        st.warning(
            "The omnibus test was **not significant**. Pairwise comparisons "
            "are shown for **exploratory purposes only** — interpret with caution."
        )
        if not st.checkbox(
            "Show exploratory pairwise comparisons anyway",
            key="posthoc_exploratory",
        ):
            return

    # Auto-suggest correction method
    posthoc_suggestion = suggest_posthoc(
        df, group_col, value_col, omnibus_test_name,
    )
    recommended = posthoc_suggestion["primary"]

    st.info(
        f"**Recommended:** {_correction_label(recommended)}  \n"
        f"{posthoc_suggestion['explanation']}"
    )

    # Correction method selector — ordered with recommendation first
    all_methods = ["bonferroni", "fdr_bh", "holm"]
    ordered_methods = [recommended] + [m for m in all_methods if m != recommended]

    correction = st.selectbox(
        "Correction method",
        options=ordered_methods,
        format_func=lambda x: (
            _correction_label(x) + (" (Recommended)" if x == recommended else "")
        ),
        key="posthoc_correction",
    )

    # Run post-hoc with selected correction, matching pairwise test to omnibus
    posthoc_df = run_posthoc_pairwise(
        df, group_col, value_col,
        correction=correction,
        omnibus_test_used=omnibus_test_name,
    )

    # Replace True/False with tick/cross icons
    sig_mask = posthoc_df["Significant"].copy()
    posthoc_df["Significant"] = posthoc_df["Significant"].apply(
        lambda x: "✔" if x else "✘"    # ✔ or ✘
    )

    # Style the table
    def _style_row(row):
        if row["Significant"] == "✔":
            return ["background-color: rgba(42, 157, 143, 0.12)"] * len(row)
        return [""] * len(row)

    def _style_sig_col(val):
        if val == "✔":
            return "color: #2A9D8F; font-weight: bold; font-size: 1.2em"
        return "color: #999999; font-size: 1.2em"

    styled = (
        posthoc_df.style
        .apply(_style_row, axis=1)
        .map(_style_sig_col, subset=["Significant"])
        .format({
            "Statistic": "{:.4f}",
            "p (raw)": "{:.6f}",
            "p (adjusted)": "{:.6f}",
            "Cohen's d": "{:.4f}",
        })
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)
    csv_data = posthoc_df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="oncoviz_posthoc.csv",
        mime="text/csv",
        key="csv_posthoc",
    )

    n_sig = sig_mask.sum()
    n_total = len(posthoc_df)
    st.caption(
        f"{n_sig} of {n_total} pairwise comparisons are significant "
        f"after {_correction_label(correction)} correction."
    )

    # Forest plot of effect sizes
    _display_forest_plot(posthoc_df, sig_mask)


def _display_forest_plot(posthoc_df: pd.DataFrame, sig_mask: pd.Series):
    """
    Forest plot showing Cohen's d with 95% CI for each pairwise comparison.
    Significant pairs are shown in teal, non-significant in grey.
    """
    layout_args = get_plotly_layout_args()

    # Build labels like "Adeno vs Large"
    labels = posthoc_df["Group A"] + " vs " + posthoc_df["Group B"]
    d_vals = posthoc_df["Cohen's d"].values
    ci_lo = posthoc_df["CI Lower"].values
    ci_hi = posthoc_df["CI Upper"].values

    # Colors: teal for significant, grey for not
    colors = ["#2A9D8F" if s else "#AAAAAA" for s in sig_mask]

    fig = go.Figure()

    # CI error bars
    for i in range(len(labels)):
        fig.add_trace(go.Scatter(
            x=[ci_lo[i], ci_hi[i]],
            y=[labels.iloc[i], labels.iloc[i]],
            mode="lines",
            line=dict(color=colors[i], width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Effect size points
    fig.add_trace(go.Scatter(
        x=d_vals,
        y=labels,
        mode="markers",
        marker=dict(
            color=colors,
            size=10,
            line=dict(color="black", width=1),
        ),
        showlegend=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Cohen's d = %{x:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Reference line at d = 0 (no effect)
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="black",
        line_width=1,
        annotation_text="No effect",
        annotation_position="top",
    )

    fig.update_layout(
        title="Forest Plot — Pairwise Effect Sizes (Cohen's d with 95% CI)",
        xaxis_title="Cohen's d",
        yaxis_title="",
        height=max(300, len(labels) * 50 + 100),
        yaxis=dict(autorange="reversed"),
        **layout_args,
    )

    st.plotly_chart(fig, use_container_width=True)
    figure_download_ui(fig, key="forest_plot", filename="oncoviz_forest")


def _correction_label(method: str) -> str:
    """Human-readable label for a correction method."""
    return {
        "bonferroni": "Bonferroni (conservative)",
        "fdr_bh": "Benjamini-Hochberg FDR (moderate)",
        "holm": "Holm-Bonferroni (step-down)",
    }.get(method, method)


# ===================================================================
# Correlation tab
# ===================================================================

def _render_correlation_tab(df: pd.DataFrame, meta: dict):
    """Compute correlation between two numerical variables."""

    st.subheader("Correlation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        col_a = st.selectbox(
            "Variable X",
            options=meta["numerical"],
            key="corr_x",
        )

    with col2:
        col_b = st.selectbox(
            "Variable Y",
            options=[c for c in meta["numerical"] if c != col_a] if col_a else meta["numerical"],
            key="corr_y",
        )

    if not col_a or not col_b:
        st.info("Select two numerical variables.")
        return

    # Show variable info
    with st.expander("Variable info"):
        st.markdown(f"**{col_a}**: {get_variable_description(col_a)}")
        st.markdown(f"**{col_b}**: {get_variable_description(col_b)}")

    # Auto-recommend method
    suggestion = suggest_correlation_method(df, col_a, col_b)
    recommended = suggestion["primary"]

    st.info(f"**Recommended:** {recommended.capitalize()}  \n{suggestion['explanation']}")

    # Method selector — recommended first
    all_methods = ["auto", "pearson", "spearman", "kendall"]
    method_labels = {
        "auto": f"Auto → {recommended.capitalize()} (Recommended)",
        "pearson": "Pearson",
        "spearman": "Spearman",
        "kendall": "Kendall's tau",
    }

    method = st.selectbox(
        "Method",
        options=all_methods,
        format_func=lambda x: method_labels[x],
        key="corr_method",
    )

    # Run — store in session_state
    if st.button("Compute Correlation", key="btn_corr", type="primary"):
        with st.spinner("Computing..."):
            try:
                result = run_correlation(df, col_a, col_b, method=method)
                st.session_state["corr_result"] = result
                st.session_state["corr_col_a"] = col_a
                st.session_state["corr_col_b"] = col_b
            except Exception as e:
                st.error(f"Error: {e}")
                return

    # Display stored results
    if "corr_result" in st.session_state:
        result = st.session_state["corr_result"]
        stored_a = st.session_state.get("corr_col_a")
        stored_b = st.session_state.get("corr_col_b")

        if stored_a == col_a and stored_b == col_b:
            _display_correlation_result(result)
            _display_scatter_plot(df, col_a, col_b, result)

    # Correlation matrix option
    st.markdown("---")
    if st.checkbox("Show full correlation matrix", key="show_corr_matrix"):
        _display_correlation_matrix(df, meta["numerical"])


# ===================================================================
# Survival analysis tab
# ===================================================================

def _render_survival_tab(df: pd.DataFrame, meta: dict, dataset_name: str):
    """Kaplan-Meier survival analysis with optional group comparison."""

    st.subheader("Survival Analysis (Kaplan-Meier)")

    info = get_dataset_info(dataset_name)
    time_col = info["time_col"]
    event_col = info["event_col"]

    time_unit = "days" if dataset_name == "VA Lung Cancer" else "months"
    st.caption(
        f"Using **{time_col}** (survival time in {time_unit}) and "
        f"**{event_col}** (event indicator: 1 = event occurred)."
    )

    # Optional grouping
    group_options = ["None (overall survival)"] + meta["categorical"] + meta["binary"]
    group_selection = st.selectbox(
        "Group by (optional)",
        options=group_options,
        key="km_group",
    )
    group_col = None if group_selection.startswith("None") else group_selection

    if group_col:
        with st.expander("Variable info"):
            st.markdown(f"**{group_col}**: {get_variable_description(group_col)}")

    if st.button("Run Survival Analysis", key="btn_survival", type="primary"):
        with st.spinner("Fitting Kaplan-Meier curves..."):
            try:
                sa = run_survival_analysis(df, time_col, event_col, group_col)
                st.session_state["survival_result"] = sa
                st.session_state["survival_group_col"] = group_col
                st.session_state["survival_time_unit"] = time_unit
            except Exception as e:
                st.error(f"Error: {e}")
                return

    # Display stored results (persists across reruns)
    if "survival_result" in st.session_state:
        sa = st.session_state["survival_result"]
        stored_group = st.session_state.get("survival_group_col")
        stored_time_unit = st.session_state.get("survival_time_unit")

        # Only show if the current group selection matches
        if stored_group == group_col:
            # Log-rank result
            if sa["log_rank"] is not None:
                _display_test_result(sa["log_rank"])

            # KM curves with censoring marks and median line
            _display_km_plot(sa["km_results"], stored_time_unit, group_col)

            # Number at risk table
            _display_at_risk_table(sa["km_results"], stored_time_unit)

            # Summary table (with median CI)
            _display_km_summary_table(sa["km_results"], stored_time_unit)


# ===================================================================
# Display helpers — test results
# ===================================================================

def _display_test_result(result: TestResult):
    """Render a test result in a structured card layout."""

    # Result card
    st.markdown("---")
    st.markdown(f"### {result.test_name}")

    col_stat, col_p, col_sig = st.columns(3)
    with col_stat:
        st.metric("Test Statistic", f"{result.statistic:.4f}")
    with col_p:
        st.metric("p-value", format_p_value(result.p_value))
    with col_sig:
        badge = significance_badge(result.p_value, result.alpha)
        st.metric("Significance", badge)

    # Interpretation
    if result.significant:
        st.success(f"**Result:** {result.interpretation}")
    else:
        st.warning(f"**Result:** {result.interpretation}")

    # Effect size details
    details = result.details
    effect_parts = []
    if "cohens_d" in details:
        effect_parts.append(
            interpret_effect_size_text("cohens_d", details["cohens_d"])
        )
    if "eta_squared" in details:
        effect_parts.append(
            interpret_effect_size_text("eta_squared", details["eta_squared"])
        )
    if "cramers_v" in details:
        effect_parts.append(
            interpret_effect_size_text("cramers_v", details["cramers_v"])
        )

    if effect_parts:
        st.caption("Effect size: " + " | ".join(effect_parts))

    # Detailed stats in expander
    with st.expander("Full details"):
        if "group_stats" in details:
            _gs_df = pd.DataFrame(details["group_stats"]).T
            dataframe_with_csv(_gs_df, key="group_stats_multi", filename="oncoviz_group_stats")
        elif "n_a" in details:
            summary = pd.DataFrame({
                details["group_a"]: {
                    "n": details["n_a"],
                    "mean": details["mean_a"],
                    "median": details["median_a"],
                    "std": details["std_a"],
                },
                details["group_b"]: {
                    "n": details["n_b"],
                    "mean": details["mean_b"],
                    "median": details["median_b"],
                    "std": details["std_b"],
                },
            })
            dataframe_with_csv(summary, key="group_stats_two", filename="oncoviz_group_stats")
        else:
            st.json(details)


# ===================================================================
# Display helpers — plots
# ===================================================================

def _display_comparison_plot(df, group_col, value_col):
    """
    Show violin plot with overlaid box plot and data points.
    - Data points are black, centered on the violin/box (overlapping).
    - Box edges, whiskers, median, and mean lines are all black.
    - Violin fill uses the group color.
    """
    plot_df = df[[group_col, value_col]].dropna()
    colors = get_chart_colors()
    layout_args = get_plotly_layout_args()

    fig = go.Figure()

    groups = sorted(plot_df[group_col].unique(), key=str)

    for i, g in enumerate(groups):
        vals = plot_df.loc[plot_df[group_col] == g, value_col]
        color = colors[i % len(colors)]

        fig.add_trace(go.Violin(
            y=vals,
            name=str(g),
            # --- Box: black edges, white fill ---
            box_visible=True,
            box_fillcolor="rgba(255,255,255,0.8)",
            box_line_color="black",
            box_line_width=1.5,
            # --- Mean line: black dashed ---
            meanline_visible=True,
            meanline_color="red",
            meanline_width=2.0,
            # --- Data points: black, centered on violin ---
            points="all",
            pointpos=0,
            jitter=0.05,
            marker=dict(
                color="black",
                size=4,
                opacity=0.6,
                line=dict(width=0),
            ),
            # --- Violin fill ---
            line_color=color,
            fillcolor=color,
            opacity=0.45,
            scalemode="width",
            width=0.75,
            showlegend=False,
        ))

    fig.update_layout(
        title=f"{value_col} by {group_col}",
        xaxis_title=group_col,
        yaxis_title=value_col,
        height=500,
        showlegend=False,
        **layout_args,
    )

    st.plotly_chart(fig, use_container_width=True)
    figure_download_ui(fig, key="comparison_violin", filename="oncoviz_comparison")


def _display_correlation_result(result: TestResult):
    """
    Render a correlation-specific result card.
    Shows r, R², 95% CI for r, and interpretation — richer than the
    generic _display_test_result used by group comparison.
    """
    st.markdown("---")
    st.markdown(f"### {result.test_name}")

    details = result.details
    r_val = details["r"]
    r_sq = details["r_squared"]
    ci_lo = details.get("ci_lower", float("nan"))
    ci_hi = details.get("ci_upper", float("nan"))

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Correlation (r)", f"{r_val:.4f}")
    with c2:
        st.metric("R²", f"{r_sq:.4f}")
    with c3:
        st.metric("p-value", format_p_value(result.p_value))
    with c4:
        badge = significance_badge(result.p_value, result.alpha)
        st.metric("Significance", badge)

    # 95% CI row
    if not (math.isnan(ci_lo) or math.isnan(ci_hi)):
        st.caption(
            f"95% CI for r: [{ci_lo:.4f}, {ci_hi:.4f}]  |  "
            f"Strength: **{details['strength']}**  |  "
            f"Direction: **{details['direction']}**  |  "
            f"n = {details['n']}"
        )
    else:
        st.caption(
            f"Strength: **{details['strength']}**  |  "
            f"Direction: **{details['direction']}**  |  "
            f"n = {details['n']}"
        )

    # Interpretation
    if result.significant:
        st.success(f"**Result:** {result.interpretation}")
    else:
        st.warning(f"**Result:** {result.interpretation}")

    # R² interpretation in expander
    with st.expander("Full details"):
        st.markdown(
            f"- **R² = {r_sq:.4f}** — {r_sq*100:.1f}% of the variance in one "
            f"variable is explained by the other.\n"
            f"- **Method:** {details['method'].capitalize()}\n"
            f"- **Sample size:** {details['n']}"
        )


def _display_scatter_plot(df, col_a, col_b, result):
    """
    Show scatter plot with trendline for correlation.
    - Pearson → OLS trendline (linear relationship).
    - Spearman / Kendall → LOWESS trendline (monotonic/non-linear).
    """
    colors = get_chart_colors()
    layout_args = get_plotly_layout_args()

    method = result.details.get("method", "pearson")
    trendline = "ols" if method == "pearson" else "lowess"
    trend_label = "OLS" if method == "pearson" else "LOWESS"

    fig = px.scatter(
        df.dropna(subset=[col_a, col_b]),
        x=col_a,
        y=col_b,
        trendline=trendline,
        title=f"{col_a} vs {col_b} (r = {result.details['r']:.3f}, {trend_label} trend)",
        color_discrete_sequence=[colors[0]],
    )
    fig.update_layout(height=450, **layout_args)
    st.plotly_chart(fig, use_container_width=True)
    figure_download_ui(fig, key="corr_scatter", filename="oncoviz_scatter")


def _display_correlation_matrix(df, numerical_cols):
    """
    Show a heatmap of the full correlation matrix.
    Uses the method stored in session_state (from the last run),
    or defaults to Spearman. Highlights strong correlations (|r| >= 0.5).
    """
    if len(numerical_cols) < 2:
        st.info("Need at least 2 numerical columns for a matrix.")
        return

    layout_args = get_plotly_layout_args()

    # Use the method from the last correlation run, default to spearman
    stored_result = st.session_state.get("corr_result")
    method = "spearman"
    if stored_result is not None:
        method = stored_result.details.get("method", "spearman")

    corr = df[numerical_cols].corr(method=method)

    # Significance filter option
    threshold = st.slider(
        "Highlight correlations with |r| ≥",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        key="corr_matrix_threshold",
    )

    # Build annotation text — mark strong correlations with asterisk
    text_matrix = []
    for i in range(len(corr)):
        row_text = []
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            marker = " *" if (abs(val) >= threshold and i != j) else ""
            row_text.append(f"{val:.2f}{marker}")
        text_matrix.append(row_text)

    fig = px.imshow(
        corr,
        text_auto=False,
        color_continuous_scale="PuOr",
        zmin=-1,
        zmax=1,
        title=f"{method.capitalize()} Correlation Matrix (* = |r| ≥ {threshold})",
        aspect="auto",
    )

    # Overlay custom text annotations
    fig.update_traces(
        text=text_matrix,
        texttemplate="%{text}",
    )

    fig.update_layout(height=500, **layout_args)
    st.plotly_chart(fig, use_container_width=True)
    figure_download_ui(fig, key="corr_matrix", filename="oncoviz_corr_matrix")

    # Count strong pairs
    strong_pairs = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                strong_pairs.append(
                    (corr.index[i], corr.columns[j], round(val, 3))
                )
    if strong_pairs:
        st.caption(
            f"**{len(strong_pairs)} pair(s)** with |r| ≥ {threshold}: "
            + ", ".join(f"{a} & {b} ({r})" for a, b, r in strong_pairs)
        )


def _display_km_plot(km_results: dict, time_unit: str, group_col: str = None):
    """
    Render Kaplan-Meier survival curves with:
    - Confidence interval bands
    - Censoring marks (+ tick marks on the curve)
    - Median survival reference line (dashed at S=0.50 with vertical drops)
    """
    fig = go.Figure()

    colors = get_chart_colors()
    layout_args = get_plotly_layout_args()

    for i, (label, km) in enumerate(km_results.items()):
        color = colors[i % len(colors)]

        # Main survival curve
        fig.add_trace(go.Scatter(
            x=km["timeline"],
            y=km["survival_prob"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=km["timeline"] + km["timeline"][::-1],
            y=km["ci_upper"] + km["ci_lower"][::-1],
            fill="toself",
            fillcolor=color,
            opacity=0.1,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name=f"{label} CI",
        ))

        # Censoring marks — small + tick marks on the curve
        censor_times = km.get("censor_times", [])
        censor_surv = km.get("censor_surv", [])
        if censor_times:
            fig.add_trace(go.Scatter(
                x=censor_times,
                y=censor_surv,
                mode="markers",
                marker=dict(
                    symbol="cross-thin",
                    size=8,
                    color=color,
                    line=dict(color=color, width=2),
                ),
                showlegend=False,
                name=f"{label} censored",
                hovertemplate=(
                    f"<b>{label} (censored)</b><br>"
                    "Time = %{x}<br>"
                    "S(t) = %{y:.3f}<extra></extra>"
                ),
            ))

        # Median survival vertical drop line
        median = km.get("median_survival")
        if median is not None:
            fig.add_trace(go.Scatter(
                x=[median, median],
                y=[0, 0.5],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Horizontal median reference line at S(t) = 0.50
    has_any_median = any(
        km.get("median_survival") is not None for km in km_results.values()
    )
    if has_any_median:
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="#888888",
            line_width=1,
            annotation_text="Median (S = 0.50)",
            annotation_position="top left",
            annotation_font_color="#888888",
        )

    title = "Kaplan-Meier Survival Curves"
    if group_col:
        title += f" by {group_col}"

    fig.update_layout(
        title=title,
        xaxis_title=f"Time ({time_unit})",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        height=550,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#DEE2E6",
            borderwidth=1,
        ),
        **layout_args,
    )
    st.plotly_chart(fig, use_container_width=True)
    figure_download_ui(fig, key="km_plot", filename="oncoviz_km_curves")


def _display_at_risk_table(km_results: dict, time_unit: str):
    """
    Show a number-at-risk table below the KM plot.
    Displays how many patients remain at evenly spaced time points.
    """
    # Determine time range across all groups
    all_times = []
    for km in km_results.values():
        at_risk = km.get("at_risk_counts", {})
        if at_risk:
            all_times.extend(at_risk.keys())

    if not all_times:
        return

    t_min, t_max = min(all_times), max(all_times)

    # Pick ~6-8 evenly spaced time points
    n_points = min(8, max(4, int((t_max - t_min) / 10)))
    time_points = np.linspace(t_min, t_max, n_points)
    time_points = [round(t, 1) for t in time_points]

    # Build the table: for each group, find the at-risk count at each time
    rows = {}
    for label, km in km_results.items():
        at_risk = km.get("at_risk_counts", {})
        if not at_risk:
            continue
        sorted_times = sorted(at_risk.keys())
        group_row = []
        for tp in time_points:
            # Find the at-risk count at or just before this time point
            count = 0
            for t in sorted_times:
                if t <= tp:
                    count = int(at_risk[t])
                else:
                    break
            group_row.append(count)
        rows[label] = group_row

    if not rows:
        return

    # Format time labels with unit
    time_labels = [f"{t:.0f} {time_unit}" for t in time_points]

    at_risk_df = pd.DataFrame(rows, index=time_labels).T
    at_risk_df.index.name = "Group"

    st.markdown(f"**Number at Risk** (time in {time_unit})")
    dataframe_with_csv(at_risk_df, key="at_risk", filename="oncoviz_at_risk")


def _display_km_summary_table(km_results: dict, time_unit: str):
    """Show a summary table of KM results per group, including median CI."""
    rows = []
    for label, km in km_results.items():
        median = km["median_survival"]
        ci_lo = km.get("median_ci_lower")
        ci_hi = km.get("median_ci_upper")

        if median is not None:
            median_str = f"{median} {time_unit}"
            if ci_lo is not None and ci_hi is not None:
                ci_str = f"[{ci_lo}, {ci_hi}]"
            else:
                ci_str = "—"
        else:
            median_str = "Not reached"
            ci_str = "—"

        rows.append({
            "Group": label,
            "N": km["n"],
            "Events": km["events"],
            f"Median Survival ({time_unit})": median_str,
            "95% CI": ci_str,
        })
    dataframe_with_csv(
        pd.DataFrame(rows), key="km_summary", filename="oncoviz_km_summary",
        hide_index=True,
    )
