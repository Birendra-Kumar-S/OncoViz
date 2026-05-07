"""Tests for plotting utilities."""

import pytest
from src.utils import plotting


def test_plotting_module_imports():
    """Verify the plotting module loads without errors."""
    assert plotting is not None
