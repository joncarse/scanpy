"""Scenario 4 - run the column-chunked HVG under a REAL dask executor.

Scenarios 1-3 use dask's default (synchronous/threaded) scheduler. Here we spin
up a ``distributed.LocalCluster`` (2 worker *processes*, 1 thread each) and run
the ``seurat_v3`` column-chunked HVG through it, mirroring how this would run on
a cluster over on-disk chunks.

What this proves:
- the same result as the in-memory computation (correctness under distribution);
- work actually executed on the workers (a non-empty task stream + registered
  workers in ``client.scheduler_info()``).

Why processes, not threads: the per-block ``clip_square_sum`` -> numba njit
kernel is compiled/executed inside workers; ``processes=True`` +
``threads_per_worker=1`` avoids the numba-in-threads crash scanpy guards against
in ``maybe_dask_process_context``.
"""

from __future__ import annotations

import pytest

import scanpy as sc
from _support import assert_hvg_close, load_lazy_csc, read_reference

pytest.importorskip("skmisc")
distributed = pytest.importorskip("distributed")

GENE_CHUNK = 200


def _run_seurat_v3(adata):
    return sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=adata.n_vars,
        inplace=False,
    )


def test_seurat_v3_under_local_cluster(hvg_store, dask_client):
    # workers are up before we compute anything
    workers = dask_client.scheduler_info()["workers"]
    assert workers, "expected the LocalCluster to have registered workers"

    adata_mem = read_reference(hvg_store)
    reference = _run_seurat_v3(adata_mem)

    adata_dask = read_reference(hvg_store)
    adata_dask.layers["counts"] = load_lazy_csc(
        hvg_store, "layers/counts", gene_chunk=GENE_CHUNK
    )
    counts = adata_dask.layers["counts"]
    assert counts.chunksize[1] != counts.shape[1]  # column-chunked input

    # Capture the task stream so we can prove work ran on the cluster.
    with distributed.get_task_stream(dask_client) as ts:
        result = _run_seurat_v3(adata_dask)

    assert len(ts.data) > 0, "no tasks executed on the distributed cluster"
    assert_hvg_close(result, reference)
