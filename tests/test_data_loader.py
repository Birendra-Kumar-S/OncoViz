"""Tests for data loading and preprocessing."""

import pytest
import pandas as pd
from src.data_loader import (
    load_dataset,
    get_dataset_names,
    get_dataset_info,
    get_column_metadata,
    get_variable_description,
)


class TestDatasetRegistry:
    """Verify dataset registry is complete and consistent."""

    def test_dataset_names_returns_list(self):
        names = get_dataset_names()
        assert isinstance(names, list)
        assert len(names) >= 2

    def test_known_datasets_present(self):
        names = get_dataset_names()
        assert "VA Lung Cancer" in names
        assert "Prostate Cancer" in names

    def test_dataset_info_has_required_keys(self):
        for name in get_dataset_names():
            info = get_dataset_info(name)
            assert "description" in info
            assert "source" in info
            assert "time_col" in info
            assert "event_col" in info


class TestLoadDataset:
    """Verify dataset loading and preprocessing."""

    def test_va_lung_loads(self):
        df = load_dataset("VA Lung Cancer")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_va_lung_shape(self):
        df = load_dataset("VA Lung Cancer")
        assert len(df) == 137
        assert len(df.columns) >= 7

    def test_va_lung_has_required_columns(self):
        df = load_dataset("VA Lung Cancer")
        for col in ["therapy", "cell", "t", "event", "kps", "age"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_va_lung_event_is_binary(self):
        df = load_dataset("VA Lung Cancer")
        assert set(df["event"].dropna().unique()).issubset({0, 1})

    def test_prostate_loads(self):
        df = load_dataset("Prostate Cancer")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 502

    def test_prostate_has_required_columns(self):
        df = load_dataset("Prostate Cancer")
        for col in ["rx", "dtime", "event", "sz", "sg", "ap"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_invalid_dataset_raises(self):
        with pytest.raises((ValueError, KeyError)):
            load_dataset("Nonexistent Dataset")


class TestColumnMetadata:
    """Verify column classification into numerical/categorical/binary."""

    def test_metadata_has_required_keys(self):
        df = load_dataset("VA Lung Cancer")
        meta = get_column_metadata(df)
        assert "numerical" in meta
        assert "categorical" in meta
        assert "binary" in meta

    def test_numerical_columns_are_numeric(self):
        df = load_dataset("VA Lung Cancer")
        meta = get_column_metadata(df)
        for col in meta["numerical"]:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"{col} classified as numerical but has dtype {df[col].dtype}"
            )

    def test_no_column_in_multiple_categories(self):
        df = load_dataset("Prostate Cancer")
        meta = get_column_metadata(df)
        all_cols = meta["numerical"] + meta["categorical"] + meta["binary"]
        assert len(all_cols) == len(set(all_cols)), "A column appears in multiple categories"


class TestVariableDescriptions:
    """Verify variable descriptions are available."""

    def test_known_variables_have_descriptions(self):
        for var in ["therapy", "cell", "t", "kps", "rx", "dtime", "sz"]:
            desc = get_variable_description(var)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_unknown_variable_returns_fallback(self):
        desc = get_variable_description("nonexistent_column_xyz")
        assert isinstance(desc, str)
