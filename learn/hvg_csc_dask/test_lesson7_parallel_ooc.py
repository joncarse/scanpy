"""Lesson 7A/7B - LocalCluster HVG on on-disk gene-block CSC zarr.

Unlike Lessons 1–6 (one CSC blob + read-time ``.rechunk``), these stores write
one CSC subgroup per gene chunk (``block_000``, …). ``load_geneblock_csc``
concatenates ``read_elem_lazy`` results so each dask column-chunk maps to a
separate on-disk group — closer to true out-of-core column parallelism.

7A: full ``pbmc3k`` (~2700 × ~32k after filter).
7B: 10x Fresh 68k PBMCs filtered matrix (~68k cells); downloads on demand unless
``HVG_LESSON7_SKIP_DOWNLOAD=1``.

Flavor: ``seurat_v3`` on ``layers/counts`` under ``LocalCluster`` processes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import zarr
from anndata import AnnData
from anndata.io import read_elem

import build_geneblock_fixtures
import scanpy as sc
from _support import (
    assert_hvg_close,
    assert_on_disk_gene_blocks,
    load_geneblock_csc,
    load_geneblock_csc_memory,
)

pytest.importorskip("skmisc")
distributed = pytest.importorskip("distributed")

HERE = Path(__file__).resolve().parent
SKIP_DOWNLOAD = os.environ.get("HVG_LESSON7_SKIP_DOWNLOAD", "") == "1"


def _meta_adata(store: Path) -> AnnData:
    root = zarr.open(str(store), mode="r")
    return AnnData(obs=read_elem(root["obs"]), var=read_elem(root["var"]))


def _run_seurat_v3(adata: AnnData):
    return sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=adata.n_vars,
        inplace=False,
    )


def _assert_parallel_hvg(store: Path, dask_client, *, full_compare: bool) -> None:
    names = assert_on_disk_gene_blocks(store, "layers/counts", min_blocks=2)
    workers = dask_client.scheduler_info()["workers"]
    assert workers, "expected the LocalCluster to have registered workers"

    adata = _meta_adata(store)
    adata.layers["counts"] = load_geneblock_csc(store, "layers/counts")
    counts = adata.layers["counts"]
    assert counts.numblocks[1] == len(names)
    assert counts.chunksize[1] != counts.shape[1]

    with distributed.get_task_stream(dask_client) as ts:
        result = _run_seurat_v3(adata)

    assert len(ts.data) > 0, "no tasks executed on the distributed cluster"
    assert np.isfinite(result["variances_norm"]).all()

    if full_compare:
        ref = _meta_adata(store)
        ref.layers["counts"] = load_geneblock_csc_memory(store, "layers/counts")
        reference = _run_seurat_v3(ref)
        assert_hvg_close(result, reference)


@pytest.mark.slow
def test_lesson7a_pbmc3k_localcluster(dask_client):
    """7A: full pbmc3k gene-block store + LocalCluster seurat_v3 vs eager hstack."""
    store = build_geneblock_fixtures.build("pbmc3k", force=False)
    _assert_parallel_hvg(store, dask_client, full_compare=True)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.skipif(
    SKIP_DOWNLOAD,
    reason="HVG_LESSON7_SKIP_DOWNLOAD=1",
)
def test_lesson7b_pbmc68k_localcluster(dask_client):
    """7B: 10x 68k gene-block store + LocalCluster seurat_v3.

    Full frame equality vs eager hstack of the same blocks (honest but heavy).
    If this becomes too slow on a given machine, fall back is layout + task stream
    + finite variances_norm only — keep ``full_compare=True`` while machines handle it.
    """
    store = build_geneblock_fixtures.build("pbmc68k", force=False)
    _assert_parallel_hvg(store, dask_client, full_compare=True)
