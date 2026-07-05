"""Consistency matrix: HVG on column-chunked CSC dask vs in-memory CSC.

Covers every flavor and both single-batch and batched modes. Combinations that
are not yet implemented are marked ``xfail(strict=True)`` so that implementing
support forces the marker to be removed.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.sparse as sps
from pandas.testing import assert_frame_equal, assert_index_equal

import scanpy as sc
from testing.scanpy._helpers.data import pbmc3k
from testing.scanpy._pytest.marks import needs
from testing.scanpy._pytest.params import ARRAY_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData

FLAVORS = ["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"]
SEURAT_V3_FLAVORS = {"seurat_v3", "seurat_v3_paper"}

CSC_DASK_PARAMS = [p for p in ARRAY_TYPES if "1d_chunked" in p.id and "csc" in p.id]


def _cases() -> list[pytest.ParameterSet]:
    cases = []
    for flavor in FLAVORS:
        for batched in (False, True):
            marks = [needs.skmisc] if flavor in SEURAT_V3_FLAVORS else []
            cases.append(
                pytest.param(
                    flavor,
                    batched,
                    marks=marks,
                    id=f"{flavor}-{'batched' if batched else 'single'}",
                )
            )
    return cases


def _make_processed_adata(flavor: str, *, batched: bool) -> AnnData:
    """Build an AnnData with a dense, preprocessed ``X`` for the given flavor."""
    if flavor in SEURAT_V3_FLAVORS:
        # matches the subset used by the existing seurat_v3 dask tests; smaller
        # slices make loess ill-conditioned on this data.
        adata = pbmc3k()[:1500, :1000].copy()
        adata.X = np.abs(np.asarray(adata.X.todense())).astype(np.float32)
    else:
        rng = np.random.default_rng(0)
        adata = sc.datasets.blobs(n_observations=120, n_variables=60, rng=rng)
        adata.X = np.abs(adata.X).astype(np.float32)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.X = np.asarray(adata.X)

    if batched:
        adata.obs["batch"] = np.tile(["a", "b"], adata.shape[0] // 2).astype(str)
    return adata


@contextmanager
def _ignore_expected_warnings():
    with warnings.catch_warnings():
        # emitted when n_top_genes exceeds the number of finite dispersions,
        # which is expected here since we deliberately select all genes.
        warnings.filterwarnings(
            "ignore",
            r"`n_top_genes`.*normalized dispersions",
            category=UserWarning,
        )
        yield


def _run_hvg(adata: AnnData, flavor: str, *, batched: bool):
    # Select every gene so the test validates the numeric consistency of the
    # per-gene statistics rather than the discrete top-N selection, whose
    # boundary can flip on sub-atol differences from dask's chunked reductions.
    # This mirrors the existing ``test_dask_consistency`` convention.
    kwargs = dict(flavor=flavor, inplace=False, n_top_genes=adata.n_vars)
    if batched:
        kwargs["batch_key"] = "batch"
    with _ignore_expected_warnings():
        return sc.pp.highly_variable_genes(adata, **kwargs)


@needs.dask
@pytest.mark.parametrize(("flavor", "batched"), _cases())
@pytest.mark.parametrize("to_dask", CSC_DASK_PARAMS)
def test_hvg_csc_dask_matches_in_memory(
    flavor: str, batched: bool, to_dask: Callable
):
    base = _make_processed_adata(flavor, batched=batched)
    dense_x = np.asarray(base.X)

    adata_mem = base.copy()
    adata_mem.X = sps.csc_matrix(dense_x)

    adata_dask = base.copy()
    adata_dask.X = to_dask(dense_x)

    out_mem = _run_hvg(adata_mem, flavor, batched=batched)
    out_dask = _run_hvg(adata_dask, flavor, batched=batched)

    assert_index_equal(out_mem.index, out_dask.index)
    assert_frame_equal(out_mem, out_dask, atol=1e-4, check_dtype=False)
