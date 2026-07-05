"""Unit tests for the dispersion-flavor helpers of `highly_variable_genes`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanpy.preprocessing._highly_variable_genes._cutoffs import _Cutoffs
from scanpy.preprocessing._highly_variable_genes._dispersion import (
    _get_disp_stats,
    _get_mean_bins,
    _nth_highest,
    _subset_genes,
)
from testing.scanpy._pytest.marks import needs


@pytest.fixture
def disp_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    means = rng.uniform(0.01, 5.0, size=40)
    dispersions = rng.uniform(0.0, 3.0, size=40)
    df = pd.DataFrame({"means": means, "dispersions": dispersions})
    return df


@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
def test_get_mean_bins_length_and_dtype(disp_df, flavor):
    bins = _get_mean_bins(disp_df["means"], flavor, n_bins=20)
    assert len(bins) == len(disp_df)
    # categories are coerced to string for efficiency
    assert bins.cat.categories.dtype == "string"


@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
def test_get_disp_stats_aligned(disp_df, flavor):
    disp_df["mean_bin"] = _get_mean_bins(disp_df["means"], flavor, n_bins=20)
    stats = _get_disp_stats(disp_df, flavor)
    assert list(stats.columns) == ["avg", "dev"]
    assert stats.index.equals(disp_df.index)


def test_get_mean_bins_bad_flavor(disp_df):
    with pytest.raises(ValueError, match="seurat.*cell_ranger"):
        _get_mean_bins(disp_df["means"], "not_a_flavor", n_bins=20)


def test_nth_highest_numpy():
    x = np.array([5.0, 1.0, 3.0, np.nan, 2.0, 4.0])
    # non-nan sorted desc: 5,4,3,2,1 -> 2nd highest is 4
    assert _nth_highest(x.copy(), 2) == 4.0


def test_nth_highest_warns_when_n_too_large():
    x = np.array([1.0, 2.0, 3.0])
    with pytest.warns(UserWarning, match="number of normalized dispersions"):
        # returns the smallest remaining value when n exceeds size
        assert _nth_highest(x.copy(), 10) == 1.0


@needs.dask
def test_nth_highest_dask_matches_numpy():
    import dask.array as da

    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    x_dask = da.from_array(x, chunks=2)
    got = _nth_highest(x_dask, 2)
    assert float(np.asarray(got)) == _nth_highest(x.copy(), 2)


class _FakeAnnData:
    def __init__(self, n_vars: int):
        self.n_vars = n_vars


def test_subset_genes_with_cutoffs():
    cutoff = _Cutoffs(min_disp=0.5, max_disp=np.inf, min_mean=0.1, max_mean=3.0)
    mean = np.array([0.2, 0.2, 4.0])
    disp = np.array([1.0, 0.1, 1.0])
    mask = _subset_genes(
        _FakeAnnData(3), mean=mean, dispersion_norm=disp, cutoff=cutoff
    )
    np.testing.assert_array_equal(mask, np.array([True, False, False]))


def test_subset_genes_with_n_top_genes():
    mean = np.array([1.0, 1.0, 1.0, 1.0])
    disp = np.array([0.1, 0.9, 0.5, 0.3])
    mask = _subset_genes(_FakeAnnData(4), mean=mean, dispersion_norm=disp, cutoff=2)
    # top-2 normalized dispersions: 0.9 and 0.5
    np.testing.assert_array_equal(mask, np.array([False, True, True, False]))
