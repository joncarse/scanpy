"""Scenario 1 - dispersion flavors on out-of-core, column-chunked CSC.

Flavors ``seurat`` and ``cell_ranger`` consume **log-normalized** values, which
the fixture stores in ``X`` (CSC). We read ``X`` lazily as a column-chunked CSC
dask array and check the result matches the fully in-memory CSC computation.

Call stack worth stepping through (set breakpoints here):

    scanpy.pp.highly_variable_genes                          # public entry
      -> _highly_variable_genes/_main.py: highly_variable_genes (dispatch)
      -> _dispersion.py: _highly_variable_genes_single_batch
           -> np.expm1(X)                 # only for flavor="seurat"
           -> fast_array_utils.stats.mean_var(X, axis=0)   # dask reduction
           -> _get_mean_bins / _get_disp_stats            # binning + z-score
           -> _subset_genes / _nth_highest                # top-N selection

The dask array is column-chunked (``chunksize[1] != shape[1]``), so ``mean_var``
runs per gene-block and is concatenated along the gene axis.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import pytest

import scanpy as sc
from _support import assert_hvg_close, load_lazy_csc, read_reference

FLAVORS = ["seurat", "cell_ranger"]
GENE_CHUNK = 200


@contextmanager
def _ignore_top_genes_warning():
    print("Ignoring top genes warning - start")
    with warnings.catch_warnings():
        # expected: we deliberately request every gene, which exceeds the number
        # of finite normalized dispersions.
        warnings.filterwarnings(
            "ignore", r"`n_top_genes`.*normalized dispersions", category=UserWarning
        )
        yield
    print("Ignored top genes warning - end")


def _run(adata, flavor: str, *, batch_key: str | None):
    print("Running _run - start")
    # Select all genes so the assertion validates the numeric per-gene stats
    # rather than a top-N boundary that can flip on sub-atol dask differences.
    with _ignore_top_genes_warning():
        return sc.pp.highly_variable_genes(
            adata,
            flavor=flavor,
            n_top_genes=adata.n_vars,
            batch_key=batch_key,
            inplace=False,
        )
    print("Running _run - end")

@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
def test_dispersion_matches_in_memory(hvg_store, flavor: str, *, batched: bool):
    print("batched: ", batched)
    print("flavor: ", flavor)
    batch_key = "batch" if batched else None

    adata_mem = read_reference(hvg_store)
    reference = _run(adata_mem, flavor, batch_key=batch_key)

    adata_dask = read_reference(hvg_store)
    adata_dask.X = load_lazy_csc(hvg_store, "X", gene_chunk=GENE_CHUNK)
    # Sanity: the input really is column-chunked CSC dask.
    assert adata_dask.X.chunksize[1] != adata_dask.X.shape[1]
    result = _run(adata_dask, flavor, batch_key=batch_key)

    assert_hvg_close(result, reference)
