"""
Statistical test implementations for OncoViz.

Every public function returns a standardised result dict with at minimum:
    - "test_name"      : str   (human-readable name of the test)
    - "statistic"      : float (test statistic value)
    - "p_value"        : float
    - "significant"    : bool  (p < alpha)
    - "interpretation" : str   (one-sentence plain-English summary)
    - "details"        : dict  (extra info specific to each test)

This makes it trivial for the UI layer to render any test result
without knowing which test was run.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Default significance level
# ---------------------------------------------------------------------------
ALPHA = 0.05


# ===================================================================
# Result container
# ===================================================================

@dataclass
class TestResult:
    """Standardised container returned by every test function."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float = ALPHA
    significant: bool = False
    interpretation: str = ""
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        self.significant = self.p_value < self.alpha

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# Normality check (used internally to pick parametric vs non-parametric)
# ===================================================================

def check_normality(series: pd.Series, alpha: float = ALPHA) -> bool:
    """
    Shapiro-Wilk test for normality.
    Returns True if the data is approximately normal (p >= alpha).
    For samples > 5000, falls back to D'Agostino-Pearson.
    """
    clean = series.dropna()
    if len(clean) < 8:
        return False
    if len(clean) > 5000:
        _, p = stats.normaltest(clean)
    else:
        _, p = stats.shapiro(clean)
    return p >= alpha


# ===================================================================
# Effect size calculators
# ===================================================================

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Calculate Cohen's d for two independent samples."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return np.nan
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


def cohens_d_ci(group_a: np.ndarray, group_b: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute approximate 95% confidence interval for Cohen's d
    using the non-central t-distribution approximation.

    Returns (d, ci_lower, ci_upper).
    """
    d = cohens_d(group_a, group_b)
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2 or np.isnan(d):
        return d, np.nan, np.nan

    # Standard error of Cohen's d
    se = np.sqrt((n_a + n_b) / (n_a * n_b) + d ** 2 / (2 * (n_a + n_b)))
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = d - z * se
    ci_upper = d + z * se
    return d, round(ci_lower, 4), round(ci_upper, 4)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if np.isnan(d_abs):
        return "unable to calculate"
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    return "large"


def eta_squared(f_stat: float, k: int, n: int) -> float:
    """
    Compute eta-squared from an ANOVA F-statistic.
    k = number of groups, n = total sample size.
    """
    if n <= k or f_stat < 0:
        return np.nan
    df_between = k - 1
    df_within = n - k
    return (f_stat * df_between) / (f_stat * df_between + df_within)


def cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    """
    Compute Cramér's V from a chi-square statistic.
    n = total observations, k = number of columns, r = number of rows.
    """
    min_dim = min(k - 1, r - 1)
    if min_dim == 0 or n == 0:
        return np.nan
    return np.sqrt(chi2 / (n * min_dim))


# ===================================================================
# Two-sample tests
# ===================================================================

def run_two_sample_test(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    test: str = "auto",
    alpha: float = ALPHA,
) -> TestResult:
    """
    Compare a continuous variable across two groups.

    Parameters
    ----------
    data : DataFrame
    group_col : str — column with exactly 2 groups
    value_col : str — continuous column to compare
    test : str — "t_test", "mann_whitney", or "auto" (checks normality)
    alpha : float

    Returns
    -------
    TestResult
    """
    df = data[[group_col, value_col]].dropna()
    groups = df[group_col].unique()

    if len(groups) != 2:
        raise ValueError(
            f"Expected 2 groups in '{group_col}', found {len(groups)}: {list(groups)}"
        )

    a = df.loc[df[group_col] == groups[0], value_col].values
    b = df.loc[df[group_col] == groups[1], value_col].values

    # Decide test type
    if test == "auto":
        normal_a = check_normality(pd.Series(a), alpha)
        normal_b = check_normality(pd.Series(b), alpha)
        test = "t_test" if (normal_a and normal_b) else "mann_whitney"

    if test == "t_test":
        # Levene's test for equal variances
        _, levene_p = stats.levene(a, b)
        equal_var = levene_p >= alpha
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
        test_label = "Independent t-test" + (" (Welch's)" if not equal_var else "")
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_label = "Mann-Whitney U test"

    d = cohens_d(a, b)
    effect_label = interpret_effect_size(d)

    result = TestResult(
        test_name=test_label,
        statistic=round(stat, 4),
        p_value=round(p, 6),
        alpha=alpha,
        interpretation=(
            f"{value_col} differs significantly between "
            f"{groups[0]} and {groups[1]} "
            f"(p = {p:.4f}, effect size: {effect_label})."
            if p < alpha
            else f"No significant difference in {value_col} between "
            f"{groups[0]} and {groups[1]} (p = {p:.4f})."
        ),
        details={
            "group_a": str(groups[0]),
            "group_b": str(groups[1]),
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": round(float(np.mean(a)), 4),
            "mean_b": round(float(np.mean(b)), 4),
            "median_a": round(float(np.median(a)), 4),
            "median_b": round(float(np.median(b)), 4),
            "std_a": round(float(np.std(a, ddof=1)), 4),
            "std_b": round(float(np.std(b, ddof=1)), 4),
            "cohens_d": round(d, 4),
            "effect_size_label": effect_label,
            "test_used": test,
        },
    )
    return result


# ===================================================================
# Multi-group tests
# ===================================================================

def run_multi_group_test(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    test: str = "auto",
    alpha: float = ALPHA,
) -> TestResult:
    """
    Compare a continuous variable across 3+ groups.

    Parameters
    ----------
    test : str — "anova", "kruskal", or "auto" (checks normality per group)
    """
    df = data[[group_col, value_col]].dropna()
    groups = df[group_col].unique()
    k = len(groups)

    if k < 2:
        raise ValueError(f"Need at least 2 groups, found {k}.")

    # If exactly 2 groups, delegate to two-sample
    if k == 2:
        return run_two_sample_test(data, group_col, value_col, test="auto", alpha=alpha)

    samples = [
        df.loc[df[group_col] == g, value_col].values for g in groups
    ]

    if test == "auto":
        all_normal = all(check_normality(pd.Series(s), alpha) for s in samples)
        test = "anova" if all_normal else "kruskal"

    if test == "anova":
        stat, p = stats.f_oneway(*samples)
        test_label = "One-way ANOVA"
        eta2 = eta_squared(stat, k, len(df))
        effect_detail = {"eta_squared": round(eta2, 4)}
    else:
        stat, p = stats.kruskal(*samples)
        test_label = "Kruskal-Wallis H test"
        # Epsilon-squared as non-parametric effect size
        n = len(df)
        eps2 = (stat - k + 1) / (n - k) if n > k else np.nan
        effect_detail = {"epsilon_squared": round(eps2, 4)}

    group_stats = {}
    for g, s in zip(groups, samples):
        group_stats[str(g)] = {
            "n": int(len(s)),
            "mean": round(float(np.mean(s)), 4),
            "median": round(float(np.median(s)), 4),
            "std": round(float(np.std(s, ddof=1)), 4),
        }

    return TestResult(
        test_name=test_label,
        statistic=round(stat, 4),
        p_value=round(p, 6),
        alpha=alpha,
        interpretation=(
            f"Significant difference in {value_col} across "
            f"{k} groups of {group_col} (p = {p:.4f})."
            if p < alpha
            else f"No significant difference in {value_col} across "
            f"{k} groups of {group_col} (p = {p:.4f})."
        ),
        details={
            "n_groups": k,
            "group_labels": [str(g) for g in groups],
            "group_stats": group_stats,
            "test_used": test,
            **effect_detail,
        },
    )


# ===================================================================
# Post-hoc pairwise comparisons with multiple testing correction
# ===================================================================

def run_posthoc_pairwise(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    correction: str = "bonferroni",
    omnibus_test_used: str = "kruskal",
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Run pairwise comparisons between all group pairs after a
    multi-group test, with multiple testing correction.

    The pairwise test matches the omnibus: t-test after ANOVA,
    Mann-Whitney U after Kruskal-Wallis.

    Parameters
    ----------
    correction : str — "bonferroni", "fdr_bh" (Benjamini-Hochberg), or "holm"
    omnibus_test_used : str — name of the omnibus test that was run

    Returns
    -------
    pd.DataFrame with columns:
        Group A, Group B, Test, Statistic, p (raw), p (adjusted),
        Significant, Cohen's d, Effect Size
    """
    df = data[[group_col, value_col]].dropna()
    groups = sorted(df[group_col].unique(), key=str)
    pairs = list(combinations(groups, 2))

    # Decide pairwise test based on omnibus
    use_parametric = omnibus_test_used in ("anova", "One-way ANOVA")

    raw_p_values = []
    rows = []

    for g_a, g_b in pairs:
        a = df.loc[df[group_col] == g_a, value_col].values
        b = df.loc[df[group_col] == g_b, value_col].values

        if use_parametric:
            # Welch's t-test (does not assume equal variances)
            stat, p = stats.ttest_ind(a, b, equal_var=False)
            test_used = "Welch's t-test"
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_used = "Mann-Whitney U"

        d, ci_lo, ci_hi = cohens_d_ci(a, b)
        raw_p_values.append(p)

        rows.append({
            "Group A": str(g_a),
            "Group B": str(g_b),
            "Test": test_used,
            "Statistic": round(stat, 4),
            "p (raw)": round(p, 6),
            "Cohen's d": round(d, 4),
            "CI Lower": ci_lo,
            "CI Upper": ci_hi,
            "Effect Size": interpret_effect_size(d),
        })

    # Apply correction
    reject, p_adj, _, _ = multipletests(raw_p_values, alpha=alpha, method=correction)

    for i, row in enumerate(rows):
        row["p (adjusted)"] = round(p_adj[i], 6)
        row["Significant"] = bool(reject[i])

    result_df = pd.DataFrame(rows)

    # Reorder columns
    result_df = result_df[[
        "Group A", "Group B", "Test", "Statistic",
        "p (raw)", "p (adjusted)", "Significant",
        "Cohen's d", "CI Lower", "CI Upper", "Effect Size",
    ]]

    return result_df


def suggest_posthoc(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    omnibus_test_used: str,
    alpha: float = ALPHA,
) -> dict:
    """
    Recommend a post-hoc correction method based on the data
    characteristics and the omnibus test that was used.

    Returns
    -------
    dict with "primary", "alternatives", and "explanation".
    """
    df = data[[group_col, value_col]].dropna()
    groups = df[group_col].unique()
    n_groups = len(groups)
    n_pairs = n_groups * (n_groups - 1) // 2

    # Check if group sizes are balanced (ratio of largest to smallest < 1.5)
    sizes = [len(df[df[group_col] == g]) for g in groups]
    ratio = max(sizes) / max(min(sizes), 1)
    balanced = ratio < 1.5

    # If non-parametric omnibus was used, or data is non-normal
    if omnibus_test_used in ("kruskal", "Kruskal-Wallis H test"):
        if n_pairs <= 6:
            return {
                "primary": "bonferroni",
                "alternatives": ["holm", "fdr_bh"],
                "explanation": (
                    f"Non-parametric omnibus test was used. With {n_pairs} "
                    f"pairwise comparisons, Bonferroni correction is recommended "
                    f"for strict family-wise error control."
                ),
            }
        else:
            return {
                "primary": "holm",
                "alternatives": ["bonferroni", "fdr_bh"],
                "explanation": (
                    f"Non-parametric omnibus test was used. With {n_pairs} "
                    f"comparisons, Holm-Bonferroni provides better power than "
                    f"plain Bonferroni while still controlling family-wise error."
                ),
            }

    # Parametric (ANOVA) — balanced groups
    if balanced and n_pairs <= 10:
        return {
            "primary": "bonferroni",
            "alternatives": ["holm", "fdr_bh"],
            "explanation": (
                f"Balanced group sizes with {n_pairs} comparisons. "
                f"Bonferroni is straightforward and widely accepted "
                f"for controlling family-wise error rate."
            ),
        }

    # Parametric — many comparisons or unbalanced
    return {
        "primary": "fdr_bh",
        "alternatives": ["holm", "bonferroni"],
        "explanation": (
            f"{'Unbalanced group sizes' if not balanced else 'Many comparisons'} "
            f"({n_pairs} pairs). Benjamini-Hochberg FDR provides better "
            f"statistical power while controlling the false discovery rate."
        ),
    }


# ===================================================================
# Chi-square test of independence
# ===================================================================

def run_chi_square(
    data: pd.DataFrame,
    col_a: str,
    col_b: str,
    alpha: float = ALPHA,
) -> TestResult:
    """
    Chi-square test of independence between two categorical variables.
    Falls back to Fisher's exact test for 2×2 tables with expected
    counts < 5.
    """
    df = data[[col_a, col_b]].dropna()
    ct = pd.crosstab(df[col_a], df[col_b])

    n = ct.values.sum()
    r, k = ct.shape

    # Check if Fisher's exact is more appropriate (2×2 with small expected)
    use_fisher = False
    if r == 2 and k == 2:
        chi2_check, _, _, expected = stats.chi2_contingency(ct)
        if (expected < 5).any():
            use_fisher = True

    if use_fisher:
        odds_ratio, p = stats.fisher_exact(ct)
        stat = odds_ratio
        test_label = "Fisher's exact test"
        extra = {"odds_ratio": round(odds_ratio, 4)}
    else:
        stat, p, dof, expected = stats.chi2_contingency(ct)
        test_label = "Chi-square test of independence"
        v = cramers_v(stat, n, k, r)
        extra = {
            "dof": int(dof),
            "cramers_v": round(v, 4),
            "expected_frequencies": expected.round(2).tolist(),
        }

    return TestResult(
        test_name=test_label,
        statistic=round(stat, 4),
        p_value=round(p, 6),
        alpha=alpha,
        interpretation=(
            f"Significant association between {col_a} and {col_b} "
            f"(p = {p:.4f})."
            if p < alpha
            else f"No significant association between {col_a} and {col_b} "
            f"(p = {p:.4f})."
        ),
        details={
            "contingency_table": ct.to_dict(),
            "n": int(n),
            **extra,
        },
    )


# ===================================================================
# Correlation
# ===================================================================

def run_correlation(
    data: pd.DataFrame,
    col_a: str,
    col_b: str,
    method: str = "pearson",
    alpha: float = ALPHA,
) -> TestResult:
    """
    Compute correlation between two numerical columns.

    Parameters
    ----------
    method : str — "pearson", "spearman", "kendall", or "auto"
                   "auto" checks normality and picks pearson or spearman.
    """
    df = data[[col_a, col_b]].dropna()

    # Auto-select method based on normality
    if method == "auto":
        normal_a = check_normality(df[col_a], alpha)
        normal_b = check_normality(df[col_b], alpha)
        method = "pearson" if (normal_a and normal_b) else "spearman"

    if method == "pearson":
        stat, p = stats.pearsonr(df[col_a], df[col_b])
        test_label = "Pearson correlation"
    elif method == "spearman":
        stat, p = stats.spearmanr(df[col_a], df[col_b])
        test_label = "Spearman rank correlation"
    elif method == "kendall":
        stat, p = stats.kendalltau(df[col_a], df[col_b])
        test_label = "Kendall's tau"
    else:
        raise ValueError(f"Unknown method '{method}'. Use pearson, spearman, kendall, or auto.")

    # 95% confidence interval for r (Fisher z-transformation)
    n = len(df)
    ci_lower, ci_upper = _correlation_ci(stat, n, alpha)

    # Interpret strength
    r_abs = abs(stat)
    if r_abs < 0.1:
        strength = "negligible"
    elif r_abs < 0.3:
        strength = "weak"
    elif r_abs < 0.5:
        strength = "moderate"
    elif r_abs < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if stat > 0 else "negative"

    return TestResult(
        test_name=test_label,
        statistic=round(stat, 4),
        p_value=round(p, 6),
        alpha=alpha,
        interpretation=(
            f"{strength.capitalize()} {direction} correlation between "
            f"{col_a} and {col_b} (r = {stat:.4f}, p = {p:.4f})."
        ),
        details={
            "r": round(stat, 4),
            "r_squared": round(stat ** 2, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n": n,
            "method": method,
            "strength": strength,
            "direction": direction,
        },
    )


def _correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    """
    Compute 95% CI for a correlation coefficient using Fisher's
    z-transformation.
    """
    if n < 4 or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z = np.arctanh(r)  # Fisher z
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return (ci_lo, ci_hi)


def suggest_correlation_method(
    data: pd.DataFrame,
    col_a: str,
    col_b: str,
    alpha: float = ALPHA,
) -> dict:
    """
    Recommend the appropriate correlation method based on normality
    of both variables.
    """
    df = data[[col_a, col_b]].dropna()
    normal_a = check_normality(df[col_a], alpha)
    normal_b = check_normality(df[col_b], alpha)

    if normal_a and normal_b:
        return {
            "primary": "pearson",
            "alternatives": ["spearman", "kendall"],
            "explanation": (
                "Both variables are approximately normally distributed. "
                "Pearson correlation measures linear association and is "
                "the most powerful test when normality holds."
            ),
        }
    else:
        reasons = []
        if not normal_a:
            reasons.append(f"{col_a} is non-normal")
        if not normal_b:
            reasons.append(f"{col_b} is non-normal")
        return {
            "primary": "spearman",
            "alternatives": ["pearson", "kendall"],
            "explanation": (
                f"{' and '.join(reasons)} (Shapiro-Wilk test). "
                f"Spearman rank correlation is recommended as it does not "
                f"assume normality and captures monotonic relationships."
            ),
        }


# ===================================================================
# Survival analysis
# ===================================================================

def run_survival_analysis(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: Optional[str] = None,
    alpha: float = ALPHA,
) -> dict:
    """
    Fit Kaplan-Meier survival curves and optionally compare groups
    with the log-rank test.

    Returns
    -------
    dict with keys:
        "km_results"  : dict mapping group_label → {
                            "timeline", "survival_prob", "ci_lower",
                            "ci_upper", "median_survival", "n", "events"
                        }
        "log_rank"    : TestResult or None (if no group_col)
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    # Deduplicate columns (e.g. group_col == event_col)
    cols = list(dict.fromkeys([time_col, event_col] + ([group_col] if group_col else [])))
    df = data[cols].dropna()

    km_results = {}

    if group_col is None:
        # Overall survival curve
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], event_observed=df[event_col], label="Overall")
        km_results["Overall"] = _extract_km(kmf)
        return {"km_results": km_results, "log_rank": None}

    # Per-group curves
    groups = sorted(df[group_col].dropna().unique(), key=str)
    for g in groups:
        mask = df[group_col] == g
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, time_col],
            event_observed=df.loc[mask, event_col],
            label=str(g),
        )
        km_results[str(g)] = _extract_km(kmf)

    # Log-rank test (pairwise if >2 groups, overall otherwise)
    if len(groups) == 2:
        mask_a = df[group_col] == groups[0]
        mask_b = df[group_col] == groups[1]
        lr = logrank_test(
            df.loc[mask_a, time_col], df.loc[mask_b, time_col],
            event_observed_A=df.loc[mask_a, event_col],
            event_observed_B=df.loc[mask_b, event_col],
        )
        log_rank_result = TestResult(
            test_name="Log-rank test",
            statistic=round(float(lr.test_statistic), 4),
            p_value=round(float(lr.p_value), 6),
            alpha=alpha,
            interpretation=(
                f"Significant difference in survival between "
                f"{groups[0]} and {groups[1]} (p = {lr.p_value:.4f})."
                if lr.p_value < alpha
                else f"No significant difference in survival between "
                f"{groups[0]} and {groups[1]} (p = {lr.p_value:.4f})."
            ),
            details={
                "groups_compared": [str(g) for g in groups],
            },
        )
    else:
        # Multivariate log-rank using lifelines multivariate_logrank_test
        from lifelines.statistics import multivariate_logrank_test
        lr = multivariate_logrank_test(
            df[time_col], df[group_col], df[event_col],
        )
        log_rank_result = TestResult(
            test_name="Log-rank test (multivariate)",
            statistic=round(float(lr.test_statistic), 4),
            p_value=round(float(lr.p_value), 6),
            alpha=alpha,
            interpretation=(
                f"Significant difference in survival across "
                f"{len(groups)} {group_col} groups (p = {lr.p_value:.4f})."
                if lr.p_value < alpha
                else f"No significant difference in survival across "
                f"{len(groups)} {group_col} groups (p = {lr.p_value:.4f})."
            ),
            details={
                "n_groups": len(groups),
                "groups_compared": [str(g) for g in groups],
            },
        )

    return {"km_results": km_results, "log_rank": log_rank_result}


def _extract_km(kmf) -> dict:
    """Extract plottable data from a fitted KaplanMeierFitter."""
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_

    # Median survival (may be NaN if curve doesn't cross 0.5)
    median = kmf.median_survival_time_
    if np.isinf(median):
        median = None
    else:
        median = round(float(median), 2)

    # Median survival 95% CI
    median_ci_lower = None
    median_ci_upper = None
    try:
        mci = kmf.confidence_interval_cumulative_density_
        # The median CI can be derived from where the CI band crosses 0.5
        # lifelines provides median_survival_time_ but not its CI directly,
        # so we compute from the survival CI columns.
        ci_lower_col = ci.iloc[:, 0]  # lower bound of survival
        ci_upper_col = ci.iloc[:, 1]  # upper bound of survival

        # Upper CI for median: last time where lower CI of survival >= 0.5
        # (lower survival bound stays above 0.5 longer → later crossing → upper bound)
        above_lower = ci_lower_col[ci_lower_col >= 0.5]
        if len(above_lower) > 0:
            median_ci_upper = round(float(above_lower.index[-1]), 2)

        # Lower CI for median: last time where upper CI of survival >= 0.5
        # (upper survival bound crosses 0.5 earlier → earlier time → lower bound)
        above_upper = ci_upper_col[ci_upper_col >= 0.5]
        if len(above_upper) > 0:
            median_ci_lower = round(float(above_upper.index[-1]), 2)

        # Ensure lower <= upper (swap if still inverted due to edge cases)
        if (median_ci_lower is not None and median_ci_upper is not None
                and median_ci_lower > median_ci_upper):
            median_ci_lower, median_ci_upper = median_ci_upper, median_ci_lower
    except Exception:
        pass

    # Censoring times and survival probabilities (for tick marks)
    # Censored = event did NOT occur at that time
    durations = kmf.durations
    observed = kmf.event_observed
    censor_times = durations[observed == 0]
    # Map each censored time to its survival probability
    censor_surv = []
    for t in censor_times:
        # Find the survival probability at or just before this time
        idx = sf.index[sf.index <= t]
        if len(idx) > 0:
            censor_surv.append(float(sf.iloc[:, 0].loc[idx[-1]]))
        else:
            censor_surv.append(1.0)

    return {
        "timeline": sf.index.tolist(),
        "survival_prob": sf.iloc[:, 0].tolist(),
        "ci_lower": ci.iloc[:, 0].tolist(),
        "ci_upper": ci.iloc[:, 1].tolist(),
        "median_survival": median,
        "median_ci_lower": median_ci_lower,
        "median_ci_upper": median_ci_upper,
        "n": int(len(kmf.event_observed)),
        "events": int(kmf.event_observed.sum()),
        "censor_times": censor_times.tolist(),
        "censor_surv": censor_surv,
        "at_risk_counts": kmf.event_table["at_risk"].to_dict(),
    }


# ===================================================================
# Smart test suggestion
# ===================================================================

def suggest_test(
    type_a: str,
    type_b: str,
    n_groups: Optional[int] = None,
) -> dict:
    """
    Recommend the appropriate statistical test(s) based on variable types.

    Parameters
    ----------
    type_a : str — "numerical", "categorical", or "binary"
    type_b : str — same
    n_groups : int or None — number of unique values in the categorical var

    Returns
    -------
    dict with "primary" (recommended test name), "alternatives" (list),
    and "explanation" (why this test was chosen).
    """
    pair = frozenset([type_a, type_b])

    # Numerical × Numerical → Correlation
    if pair == frozenset(["numerical"]):
        return {
            "primary": "pearson",
            "alternatives": ["spearman", "kendall"],
            "test_function": "run_correlation",
            "explanation": (
                "Both variables are continuous — correlation measures "
                "the strength and direction of their linear relationship. "
                "If the data is non-normal or contains outliers, Spearman "
                "rank correlation is more robust."
            ),
        }

    # Categorical/Binary × Numerical → Group comparison
    if ("numerical" in pair) and (pair & {"categorical", "binary"}):
        if n_groups is not None and n_groups == 2:
            return {
                "primary": "auto (t-test / Mann-Whitney)",
                "alternatives": ["t_test", "mann_whitney"],
                "test_function": "run_two_sample_test",
                "explanation": (
                    "Comparing a continuous variable across 2 groups. "
                    "If the data is approximately normal, an independent "
                    "t-test is used; otherwise the non-parametric "
                    "Mann-Whitney U test is applied."
                ),
            }
        else:
            return {
                "primary": "auto (ANOVA / Kruskal-Wallis)",
                "alternatives": ["anova", "kruskal"],
                "test_function": "run_multi_group_test",
                "explanation": (
                    "Comparing a continuous variable across 3+ groups. "
                    "If all groups are approximately normal, one-way ANOVA "
                    "is used; otherwise the non-parametric Kruskal-Wallis test."
                ),
            }

    # Categorical × Categorical (or Binary × Binary/Categorical)
    if pair <= {"categorical", "binary"}:
        return {
            "primary": "chi_square",
            "alternatives": ["fisher_exact"],
            "test_function": "run_chi_square",
            "explanation": (
                "Both variables are categorical — the chi-square test "
                "checks whether they are independent. For small 2×2 tables, "
                "Fisher's exact test is used instead."
            ),
        }

    # Fallback
    return {
        "primary": "unknown",
        "alternatives": [],
        "test_function": None,
        "explanation": "Could not determine an appropriate test for this variable combination.",
    }
