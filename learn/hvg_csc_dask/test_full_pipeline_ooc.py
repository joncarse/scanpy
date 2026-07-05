"""Scenario 3 - the realistic pipeline, end to end, on out-of-core data.

A user rarely calls HVG in isolation. Here we start from lazy raw counts on disk
and run the whole ``normalize_total`` -> ``log1p`` -> ``highly_variable_genes``
pipeline, comparing against the identical fully in-memory pipeline.

Layout note (worth a breakpoint at
``preprocessing/_normalization.py`` around line 257):

    if isinstance(x, CSCBase):
        x = x.tocsr()

- On an **in-memory scipy** matrix, ``normalize_total`` converts CSC -> CSR, so
  the layout entering ``log1p``/HVG is CSR (verified in this test).
- On a **dask** array, that scipy branch does not fire the same way: the array
  keeps its CSC ``_meta`` and its ``(-1, gene_chunk)`` column chunking all the
  way into HVG (also verified here).

Either way the final HVG statistics match, which is the point: the tutorial shows
the layout can change mid-pipeline without changing results.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import scanpy as sc
from anndata.abc import CSCDataset  # noqa: F401  (kept for discoverability)
from scipy.sparse import csr_matrix

from _support import assert_hvg_close, load_lazy_csc, read_reference

GENE_CHUNK = 200
TARGET_SUM = 1e4


@contextmanager
def _ignore_top_genes_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", r"`n_top_genes`.*normalized dispersions", category=UserWarning
        )
        yield


def _pipeline(adata):
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)
    with _ignore_top_genes_warning():
        return sc.pp.highly_variable_genes(
            adata, flavor="seurat", n_top_genes=adata.n_vars, inplace=False
        )


def test_full_pipeline_matches_in_memory(hvg_store):
    # In-memory reference: counts (CSC) -> normalize_total flips to CSR.
    adata_mem = read_reference(hvg_store)
    adata_mem.X = adata_mem.layers["counts"].copy()
    sc.pp.normalize_total(adata_mem, target_sum=TARGET_SUM)
    assert isinstance(adata_mem.X, csr_matrix)  # documented layout transition
    sc.pp.log1p(adata_mem)
    with _ignore_top_genes_warning():
        reference = sc.pp.highly_variable_genes(
            adata_mem, flavor="seurat", n_top_genes=adata_mem.n_vars, inplace=False
        )

    # Out-of-core: start from lazy, column-chunked CSC counts.
    adata_dask = read_reference(hvg_store)
    adata_dask.X = load_lazy_csc(hvg_store, "layers/counts", gene_chunk=GENE_CHUNK)
    assert getattr(adata_dask.X._meta, "format", None) == "csc"
    sc.pp.normalize_total(adata_dask, target_sum=TARGET_SUM)
    # dask keeps CSC + column chunking (unlike the scipy CSC->CSR flip above).
    assert getattr(adata_dask.X._meta, "format", None) == "csc"
    assert adata_dask.X.chunksize[1] != adata_dask.X.shape[1]
    sc.pp.log1p(adata_dask)
    with _ignore_top_genes_warning():
        result = sc.pp.highly_variable_genes(
            adata_dask, flavor="seurat", n_top_genes=adata_dask.n_vars, inplace=False
        )

    assert_hvg_close(result, reference)
