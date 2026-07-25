"""Lesson 7A/7B - LocalCluster HVG on on-disk gene-block CSC zarr.

Unlike Lessons 1–6 (one CSC blob + read-time ``.rechunk``), these stores write
one CSC subgroup per gene chunk (``block_000``, …). ``load_geneblock_csc``
concatenates ``read_elem_lazy`` results so each dask column-chunk maps to a
separate on-disk group — closer to true out-of-core column parallelism.

7A: full ``pbmc3k`` (~2700 × ~32k after filter).
7B: 10x Fresh 68k PBMCs filtered matrix (~68k cells); downloads on demand unless
``HVG_LESSON7_SKIP_DOWNLOAD=1``.

Flavor: ``seurat_v3`` on ``layers/counts`` under ``LocalCluster`` processes.

Story logs: set ``HVG_LESSON7_NARRATE=1`` and run pytest with ``-s``.
"""

from __future__ import annotations

import os
from pathlib import Path

import distributed  # hard requirement for Lesson 7 LocalCluster
import numpy as np
import pytest
import skmisc  # noqa: F401  # hard requirement: seurat_v3 loess
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
    narrate,
)

HERE = Path(__file__).resolve().parent
SKIP_DOWNLOAD = os.environ.get("HVG_LESSON7_SKIP_DOWNLOAD", "") == "1"


def _meta_adata(store: Path) -> AnnData:
    narrate("client", "_meta_adata: open zarr for obs/var only", store=str(store))
    root = zarr.open(str(store), mode="r")
    # Gene-block stores are not a full AnnData layout for X/layers.
    obs = read_elem(root["obs"])
    var = read_elem(root["var"])
    narrate(
        "client",
        "_meta_adata: loaded obs/var metadata",
        n_obs=len(obs),
        n_vars=len(var),
    )
    return AnnData(obs=obs, var=var)


def _run_seurat_v3(adata: AnnData):
    narrate(
        "client",
        "_run_seurat_v3: calling sc.pp.highly_variable_genes",
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=adata.n_vars,
        inplace=False,
        shape=adata.shape,
        counts_type=type(adata.layers.get("counts")).__name__,
    )
    result = sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=adata.n_vars,
        inplace=False,
    )
    narrate(
        "client",
        "_run_seurat_v3: got result DataFrame",
        result_shape=getattr(result, "shape", None),
        columns=list(result.columns) if result is not None else None,
    )
    return result


def _assert_parallel_hvg(store: Path, dask_client, *, full_compare: bool) -> None:
    narrate(
        "client",
        "_assert_parallel_hvg: begin",
        store=str(store),
        full_compare=full_compare,
    )
    names = assert_on_disk_gene_blocks(store, "layers/counts", min_blocks=2)
    workers = dask_client.scheduler_info()["workers"]
    narrate(
        "client",
        "_assert_parallel_hvg: scheduler workers",
        n_workers=len(workers),
        addresses=list(workers),
    )
    assert workers, "expected the LocalCluster to have registered workers"

    adata = _meta_adata(store)
    # Attach lazy column-chunked counts (Lesson 7 layout).
    adata.layers["counts"] = load_geneblock_csc(store, "layers/counts")
    counts = adata.layers["counts"]
    narrate(
        "client",
        "_assert_parallel_hvg: counts layer attached",
        shape=counts.shape,
        numblocks=counts.numblocks,
        chunksize=counts.chunksize,
        n_disk_blocks=len(names),
    )
    assert counts.numblocks[1] == len(names)
    assert counts.chunksize[1] != counts.shape[1]
    narrate(
        "client",
        "_assert_parallel_hvg: chunking assertions passed — next run HVG under task stream",
    )

    # Task stream proves work ran on the distributed cluster (not only client).
    with distributed.get_task_stream(dask_client) as ts:
        narrate("client", "_assert_parallel_hvg: task stream recording ON")
        result = _run_seurat_v3(adata)
    narrate(
        "client",
        "_assert_parallel_hvg: task stream recording OFF",
        n_tasks=len(ts.data),
    )

    assert len(ts.data) > 0, "no tasks executed on the distributed cluster"
    assert np.isfinite(result["variances_norm"]).all()
    narrate(
        "client",
        "_assert_parallel_hvg: HVG result finite",
        n_hvg=int(result["highly_variable"].sum()),
        variances_norm_finite=bool(np.isfinite(result["variances_norm"]).all()),
    )

    if full_compare:
        narrate("compare", "full_compare: eager reference HVG starting")
        ref = _meta_adata(store)
        ref.layers["counts"] = load_geneblock_csc_memory(store, "layers/counts")
        narrate(
            "compare",
            "full_compare: reference counts are in-memory scipy CSC",
            type=type(ref.layers["counts"]).__name__,
            shape=ref.layers["counts"].shape,
        )
        reference = _run_seurat_v3(ref)
        assert_hvg_close(result, reference)
        narrate("compare", "full_compare: dask vs eager HVG frames match")


@pytest.mark.slow
def test_lesson7a_pbmc3k_localcluster(dask_client):
    """7A: full pbmc3k gene-block store + LocalCluster seurat_v3 vs eager hstack."""
    narrate("client", "test_lesson7a: build/reuse pbmc3k geneblock store")
    store = build_geneblock_fixtures.build("pbmc3k", force=False)
    narrate("client", "test_lesson7a: store ready", store=str(store))
    _assert_parallel_hvg(store, dask_client, full_compare=True)
    narrate("client", "test_lesson7a: PASSED")


def test_lesson7b_pbmc68k_localcluster(dask_client):
    """7B: 10x 68k gene-block store + LocalCluster seurat_v3."""
    narrate("client", "test_lesson7b: build/reuse pbmc68k geneblock store")
    store = build_geneblock_fixtures.build("pbmc68k", force=False)
    narrate("client", "test_lesson7b: store ready", store=str(store))
    _assert_parallel_hvg(store, dask_client, full_compare=True)
    narrate("client", "test_lesson7b: PASSED")
