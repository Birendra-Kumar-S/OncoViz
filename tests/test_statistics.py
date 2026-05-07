"""Tests for statistical analysis functions."""

import pytest
import numpy as np
import pandas as pd
from src.utils.statistics import (
    TestResult,
    check_normality,
    cohens_d,
    cohens_d_ci,
    interpret_effect_size,
    eta_squared,
    cramers_v,
    run_two_sample_test,
    run_multi_group_test,
    run_posthoc_pairwise,
    run_correlation,
    run_chi_square,
    suggest_test,
    suggest_correlation_method,
    suggest_posthoc,
)


# ── Fixtures ──

@pytest.fixture
def normal_data():
    """Two normally distributed groups with known different means."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "group": ["A"] * 100 + ["B"] * 100,
        "value": np.concatenate([rng.normal(10, 2, 100), rng.normal(18, 2, 100)]),
    })
    return df


@pytest.fixture
def three_group_data():
    """Three groups for multi-group tests."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "group": ["X"] * 80 + ["Y"] * 80 + ["Z"] * 80,
        "value": np.concatenate([
            rng.normal(10, 2, 80),
            rng.normal(16, 2, 80),
            rng.normal(22, 2, 80),
        ]),
    })
    return df


@pytest.fixture
def corr_data():
    """Two correlated numerical variables."""
    rng = np.random.RandomState(42)
    x = rng.normal(0, 1, 200)
    y = 3 * x + rng.normal(0, 0.3, 200)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def categorical_data():
    """Two categorical variables for chi-square."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "treatment": rng.choice(["Drug", "Placebo"], 200),
        "outcome": rng.choice(["Improved", "No change", "Worsened"], 200),
    })
    return df


# ── TestResult ──

class TestTestResult:
    def test_significance_auto_set(self):
        r = TestResult(test_name="test", statistic=1.0, p_value=0.01)
        assert r.significant

    def test_not_significant(self):
        r = TestResult(test_name="test", statistic=1.0, p_value=0.10)
        assert not r.significant

    def test_to_dict(self):
        r = TestResult(test_name="test", statistic=1.0, p_value=0.03)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["test_name"] == "test"
        assert d["significant"]


# ── Normality Check ──

class TestNormality:
    def test_normal_data_passes(self):
        rng = np.random.RandomState(0)
        assert check_normality(pd.Series(rng.normal(0, 1, 500))) == True

    def test_skewed_data_fails(self):
        rng = np.random.RandomState(0)
        assert check_normality(pd.Series(rng.exponential(1, 500))) == False

    def test_small_sample_returns_false(self):
        assert check_normality(pd.Series([1, 2, 3])) == False


# ── Effect Sizes ──

class TestEffectSizes:
    def test_cohens_d_identical_groups(self):
        a = np.array([1, 2, 3, 4, 5])
        assert cohens_d(a, a) == 0.0

    def test_cohens_d_different_groups(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        d = cohens_d(a, b)
        assert d < 0  # a < b → negative d
        assert abs(d) > 2.0  # large effect

    def test_cohens_d_ci_returns_tuple(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        d, lo, hi = cohens_d_ci(a, b)
        assert lo < d < hi

    def test_interpret_effect_size_ranges(self):
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(0.3) == "small"
        assert interpret_effect_size(0.6) == "medium"
        assert interpret_effect_size(1.0) == "large"

    def test_eta_squared_positive(self):
        result = eta_squared(5.0, 3, 60)
        assert 0 < result < 1

    def test_cramers_v_range(self):
        v = cramers_v(10.0, 100, 3, 3)
        assert 0 <= v <= 1


# ── Two-Sample Tests ──

class TestTwoSampleTest:
    def test_returns_test_result(self, normal_data):
        result = run_two_sample_test(normal_data, "group", "value")
        assert isinstance(result, TestResult)

    def test_detects_significant_difference(self, normal_data):
        result = run_two_sample_test(normal_data, "group", "value", test="t_test")
        assert result.significant
        assert result.p_value < 0.05

    def test_auto_selects_test(self, normal_data):
        result = run_two_sample_test(normal_data, "group", "value", test="auto")
        assert result.test_name in [
            "Independent t-test",
            "Independent t-test (Welch's)",
            "Mann-Whitney U test",
        ]

    def test_explicit_t_test(self, normal_data):
        result = run_two_sample_test(normal_data, "group", "value", test="t_test")
        assert "t-test" in result.test_name

    def test_explicit_mann_whitney(self, normal_data):
        result = run_two_sample_test(normal_data, "group", "value", test="mann_whitney")
        assert result.test_name == "Mann-Whitney U test"

    def test_details_has_cohens_d(self, normal_data):
        result = run_two_sample_test(normal_data, "group", "value")
        assert "cohens_d" in result.details

    def test_wrong_group_count_raises(self):
        df = pd.DataFrame({"group": ["A", "B", "C"], "value": [1, 2, 3]})
        with pytest.raises(ValueError):
            run_two_sample_test(df, "group", "value")


# ── Multi-Group Tests ──

class TestMultiGroupTest:
    def test_returns_test_result(self, three_group_data):
        result = run_multi_group_test(three_group_data, "group", "value")
        assert isinstance(result, TestResult)

    def test_detects_significant_difference(self, three_group_data):
        result = run_multi_group_test(three_group_data, "group", "value", test="anova")
        assert result.significant

    def test_auto_selects_anova_or_kruskal(self, three_group_data):
        result = run_multi_group_test(three_group_data, "group", "value", test="auto")
        assert result.test_name in ["One-way ANOVA", "Kruskal-Wallis H test"]

    def test_has_effect_size(self, three_group_data):
        result = run_multi_group_test(three_group_data, "group", "value")
        d = result.details
        assert "eta_squared" in d or "epsilon_squared" in d


# ── Post-Hoc Pairwise ──

class TestPostHoc:
    def test_returns_dataframe(self, three_group_data):
        df = run_posthoc_pairwise(
            three_group_data, "group", "value",
            correction="bonferroni", omnibus_test_used="anova",
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 choose 2

    def test_has_required_columns(self, three_group_data):
        df = run_posthoc_pairwise(
            three_group_data, "group", "value",
            correction="bonferroni", omnibus_test_used="anova",
        )
        for col in ["Group A", "Group B", "p (raw)", "p (adjusted)", "Significant", "Cohen's d"]:
            assert col in df.columns

    def test_adjusted_p_geq_raw_p(self, three_group_data):
        df = run_posthoc_pairwise(
            three_group_data, "group", "value",
            correction="bonferroni", omnibus_test_used="anova",
        )
        for _, row in df.iterrows():
            assert row["p (adjusted)"] >= row["p (raw)"] - 1e-10

    def test_fdr_correction(self, three_group_data):
        df = run_posthoc_pairwise(
            three_group_data, "group", "value",
            correction="fdr_bh", omnibus_test_used="kruskal",
        )
        assert len(df) == 3


# ── Correlation ──

class TestCorrelation:
    def test_strong_positive_correlation(self, corr_data):
        result = run_correlation(corr_data, "x", "y", method="pearson")
        assert result.details["r"] > 0.9
        assert result.significant

    def test_r_squared_computed(self, corr_data):
        result = run_correlation(corr_data, "x", "y", method="pearson")
        r = result.details["r"]
        r2 = result.details["r_squared"]
        assert abs(r2 - r ** 2) < 0.001

    def test_ci_contains_r(self, corr_data):
        result = run_correlation(corr_data, "x", "y", method="pearson")
        r = result.details["r"]
        assert result.details["ci_lower"] < r < result.details["ci_upper"]

    def test_spearman_works(self, corr_data):
        result = run_correlation(corr_data, "x", "y", method="spearman")
        assert result.test_name == "Spearman rank correlation"
        assert result.details["r"] > 0.8

    def test_kendall_works(self, corr_data):
        result = run_correlation(corr_data, "x", "y", method="kendall")
        assert result.test_name == "Kendall's tau"

    def test_auto_selects_method(self, corr_data):
        result = run_correlation(corr_data, "x", "y", method="auto")
        assert result.test_name in [
            "Pearson correlation",
            "Spearman rank correlation",
        ]

    def test_invalid_method_raises(self, corr_data):
        with pytest.raises(ValueError):
            run_correlation(corr_data, "x", "y", method="invalid")


# ── Chi-Square ──

class TestChiSquare:
    def test_returns_test_result(self, categorical_data):
        result = run_chi_square(categorical_data, "treatment", "outcome")
        assert isinstance(result, TestResult)

    def test_has_contingency_table(self, categorical_data):
        result = run_chi_square(categorical_data, "treatment", "outcome")
        assert "contingency_table" in result.details


# ── Suggestion Engines ──

class TestSuggestions:
    def test_suggest_test_num_num(self):
        result = suggest_test("numerical", "numerical")
        assert result["primary"] == "pearson"

    def test_suggest_test_cat_num_two_groups(self):
        result = suggest_test("categorical", "numerical", n_groups=2)
        assert "t-test" in result["primary"] or "Mann-Whitney" in result["primary"]

    def test_suggest_test_cat_num_multi_groups(self):
        result = suggest_test("categorical", "numerical", n_groups=4)
        assert "ANOVA" in result["primary"] or "Kruskal" in result["primary"]

    def test_suggest_test_cat_cat(self):
        result = suggest_test("categorical", "categorical")
        assert result["primary"] == "chi_square"

    def test_suggest_correlation_method(self, corr_data):
        result = suggest_correlation_method(corr_data, "x", "y")
        assert result["primary"] in ["pearson", "spearman"]
        assert "explanation" in result

    def test_suggest_posthoc(self, three_group_data):
        result = suggest_posthoc(
            three_group_data, "group", "value",
            omnibus_test_used="anova",
        )
        assert result["primary"] in ["bonferroni", "holm", "fdr_bh"]
        assert "explanation" in result
