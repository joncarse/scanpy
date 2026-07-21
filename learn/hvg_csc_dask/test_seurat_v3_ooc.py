"""Scenario 2 - seurat_v3 flavors on out-of-core, column-chunked CSC counts.

``seurat_v3`` / ``seurat_v3_paper`` consume **raw counts**, stored by the fixture
in ``layers["counts"]`` (CSC). We read that layer lazily as a column-chunked CSC
dask array and compare single-batch and batched results to the in-memory CSC run.

Call stack worth stepping through (set breakpoints here):

    scanpy.pp.highly_variable_genes(flavor="seurat_v3", layer="counts")
      -> _seurat_v3.py: _highly_variable_genes_seurat_v3
           -> _raise_if_unsupported_dask_chunking(data)   # allows CSC col-chunk
           -> stats.mean_var(data, axis=0)                 # dask reduction
           -> per batch:
                aggregate(..., func=["mean","var"])        # get/_aggregated.py
                  -> aggregate_dask (column-chunked branch)
                loess(...).fit()                           # skmisc
                clip_square_sum(data_batch, clip_val)      # singledispatch
                  -> DaskArray branch, chunksize[1] != shape[1]:
                     _clip_square_sum_feature_chunked       # NEW code path
           -> argsort ranks -> highly_variable selection

Batched mode additionally exercises ``data[batch_info == b]`` row-masking on a
column-chunked dask array.
"""

from __future__ import annotations

import pytest

import scanpy as sc
from _support import assert_hvg_close, load_lazy_csc, read_reference

pytest.importorskip("skmisc")

FLAVORS = ["seurat_v3", "seurat_v3_paper"]
GENE_CHUNK = 200


def _run(adata, flavor: str, *, batch_key: str | None):
    # n_top_genes=n_vars -> every gene selected, so we compare the numeric
    # rank/variance columns instead of a flip-prone top-N boundary.
    return sc.pp.highly_variable_genes(
        adata,
        flavor=flavor,
        layer="counts",
        n_top_genes=adata.n_vars,
        batch_key=batch_key,
        inplace=False,
    )


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
def test_seurat_v3_matches_in_memory(hvg_store, flavor: str, *, batched: bool):
    batch_key = "batch" if batched else None

    adata_mem = read_reference(hvg_store)
    reference = _run(adata_mem, flavor, batch_key=batch_key)

    adata_dask = read_reference(hvg_store)
    adata_dask.layers["counts"] = load_lazy_csc(
        hvg_store, "layers/counts", gene_chunk=GENE_CHUNK
    )
    counts = adata_dask.layers["counts"]
    # Sanity: the input really is column-chunked CSC dask.
    assert counts.chunksize[1] != counts.shape[1]
    assert counts.chunksize[0] == counts.shape[0]  # observation axis whole
    result = _run(adata_dask, flavor, batch_key=batch_key)

    assert_hvg_close(result, reference)
