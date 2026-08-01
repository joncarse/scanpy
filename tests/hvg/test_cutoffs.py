"""Unit tests for the `_cutoffs` submodule of `highly_variable_genes`."""

from __future__ import annotations

import numpy as np
import pytest

from scanpy.preprocessing._highly_variable_genes._cutoffs import _Cutoffs


def test_validate_returns_cutoffs_without_n_top_genes():
    cutoff = _Cutoffs.validate(
        n_top_genes=None, min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3
    )
    assert isinstance(cutoff, _Cutoffs)
    assert cutoff.min_disp == 0.5
    assert cutoff.max_mean == 3


def test_validate_returns_int_with_default_cutoffs():
    cutoff = _Cutoffs.validate(
        n_top_genes=10, min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3
    )
    assert cutoff == 10


def test_validate_warns_when_cutoffs_and_n_top_genes():
    with pytest.warns(UserWarning, match="all cutoffs are ignored"):
        cutoff = _Cutoffs.validate(
            n_top_genes=10, min_disp=0.9, max_disp=np.inf, min_mean=0.0125, max_mean=3
        )
    assert cutoff == 10


def test_in_bounds():
    cutoff = _Cutoffs(min_disp=0.5, max_disp=2.0, min_mean=0.1, max_mean=3.0)
    mean = np.array([0.05, 0.2, 0.2, 4.0])
    disp = np.array([1.0, 0.4, 1.0, 1.0])
    # gene 0 fails min_mean, gene 1 fails min_disp, gene 2 in bounds, gene 3 fails max_mean
    np.testing.assert_array_equal(
        cutoff.in_bounds(mean, disp), np.array([False, False, True, False])
    )
