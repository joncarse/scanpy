"""Lesson 7A/7B - LocalCluster HVG on on-disk gene-block CSC zarr.

Unlike Lessons 1–6 (one CSC blob + read-time ``.rechunk``), these stores write
one CSC subgroup per gene chunk (``block_000``, …). ``load_geneblock_csc``
concatenates ``read_elem_lazy`` results so each dask column-chunk maps to a
separate on-disk group — closer to true out-of-core column parallelism.

7A: full ``pbmc3k`` (~2700 × ~32k after filter).
7B: 10x Fresh 68k PBMCs filtered matrix (~68k cells); downloads on demand unless
``HVG_LESSON7_SKIP_DOWNLOAD=1``.

Flavor: ``seurat_v3`` on ``layers/counts`` under ``LocalCluster`` processes.

Lecture notes live in ``# L7-LECTURE`` comments (prose + captured 7B values).
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
)

HERE = Path(__file__).resolve().parent
SKIP_DOWNLOAD = os.environ.get("HVG_LESSON7_SKIP_DOWNLOAD", "") == "1"


def _meta_adata(store: Path) -> AnnData:
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_meta_adata  where=client  pid=2237068  hits_at_site=2
    # topic: We need an AnnData object because scanpy's highly_variable_genes API expects one.
    # --- lecture ---
    # We need an AnnData object because scanpy's highly_variable_genes API
    # expects one. But our Lesson 7 Zarr store is NOT a normal full AnnData
    # layout for the expression matrices: counts live under
    # layers/counts/block_*, not under a single layers/counts array.
    #
    # So we only read the small metadata tables — obs (per-cell annotations)
    # and var (per-gene annotations / gene names). That gives us the correct
    # number of cells and genes and the right index labels, with almost no
    # RAM use.
    #
    # In a moment we will attach the counts layer ourselves by calling
    # load_geneblock_csc, which understands the block_* layout.
    # --- facts at this step ---
    #   store = '/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr'
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    root = zarr.open(str(store), mode="r")
    obs = read_elem(root["obs"])
    var = read_elem(root["var"])
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_meta_adata  where=client  pid=2237068  hits_at_site=2
    # topic: obs and var are now ordinary pandas-like tables in memory.
    # --- lecture ---
    # obs and var are now ordinary pandas-like tables in memory. n_obs is how
    # many cells; n_vars is how many genes. For Lesson 7B (pbmc68k gene-block
    # fixture) after the builder's filtering that is 68579 × 20387 — not the
    # raw unfiltered 10x gene universe (~32738 genes).
    #
    # The DataFrame *shapes* alone hide the column meanings. Those columns
    # were written by build_geneblock_fixtures._prepare:
    #
    #   obs shape (68579, 1) — one column, 'batch'
    #     • dtype category with categories ['a', 'b']
    #     • values alternate a, b, a, b, ... across cells
    #       (np.tile(["a", "b"], n_obs) in the builder)
    #     • value_counts ≈ a: 34290, b: 34289
    #     • index = 10x cell barcodes (e.g. AAACATACACCCAA-1, ...)
    #
    #   var shape (20387, 2) — columns ['gene_ids', 'n_cells']
    #     • gene_ids: Ensembl IDs from the 10x mtx (e.g. ENSG00000237683)
    #     • n_cells: how many cells detected the gene (from filter_genes)
    #     • index = gene symbols (e.g. AL627309.1, AP006222.2, ...)
    #
    # The AnnData we build next is a thin shell: this metadata present,
    # expression matrices still missing until we assign layers['counts'].
    # --- facts at this step ---
    #   n_obs = 68579
    #   n_vars = 20387
    #   obs.columns = ['batch']   # category a/b, alternating
    #   var.columns = ['gene_ids', 'n_cells']
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   root = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
    #   obs = DataFrame shape (68579, 1), columns=['batch'], batch head=['a','b','a','b',...]
    #   var = DataFrame shape (20387, 2), columns=['gene_ids', 'n_cells']
    return AnnData(obs=obs, var=var)


def _run_seurat_v3(adata: AnnData):
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_run_seurat_v3  where=client  pid=2237068  hits_at_site=2
    # topic: Now we call scanpy's public API for highly variable genes with the Seurat v3 flavor.
    # --- lecture ---
    # Now we call scanpy's public API for highly variable genes with the
    # Seurat v3 flavor. That flavor expects raw counts (not log-normalized
    # data), which is why we point it at layer='counts'.
    #
    # n_top_genes=adata.n_vars means "rank every gene and mark all of them
    # in the table" for comparison purposes — we are not subsetting to the
    # usual 2000 HVGs here, because the test wants a full comparable frame.
    #
    # inplace=False means: do not write results into adata.var; return a
    # pandas DataFrame instead. That DataFrame is one row per gene with
    # columns like means, variances, variances_norm, highly_variable, …
    # It is NOT a cells×genes expression matrix.
    # --- facts at this step ---
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   inplace = False
    #   shape = (68579, 20387)
    #   counts_type = 'Array'
    # --- locals / object fields at the call site ---
    #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
    result = sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=adata.n_vars,
        inplace=False,
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_run_seurat_v3  where=client  pid=2237068  hits_at_site=2
    # topic: highly_variable_genes returned.
    # --- lecture ---
    # highly_variable_genes returned. If counts were a dask array, a lot of
    # work just happened on the LocalCluster workers (mean/var and the
    # clipped sum-of-squares stages). If counts were an in-memory scipy
    # matrix, the same math ran in this process without shipping tasks.
    #
    # The result table's rows are genes. Later we will check that
    # variances_norm is finite and, in full_compare mode, that it matches
    # the eager reference run.
    # --- facts at this step ---
    #   result_shape = (20387, 6)
    #   columns = ['means', 'variances', 'gene_name', 'highly_variable_rank', 'variances_norm', 'highly_variable']
    # --- locals / object fields at the call site ---
    #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
    #   result = {'type': 'DataFrame', 'shape': (20387, 6)}
    return result


def _assert_parallel_hvg(store: Path, dask_client, *, full_compare: bool) -> None:
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
    # topic: This function is the Lesson 7 exam script.
    # --- lecture ---
    # This function is the Lesson 7 exam script. It will:
    #
    # 1) prove the Zarr store has multiple on-disk gene blocks,
    # 2) prove the LocalCluster has workers,
    # 3) build a lazy column-chunked counts matrix from those blocks,
    # 4) run seurat_v3 while recording the Dask task stream,
    # 5) optionally repeat with an in-memory matrix and compare answers.
    #
    # Keep the process model in your head the whole time: pytest/client
    # builds graphs and waits; workers execute chunk tasks.
    # --- facts at this step ---
    #   store = '/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr'
    #   full_compare = True
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   full_compare = True
    names = assert_on_disk_gene_blocks(store, "layers/counts", min_blocks=2)
    workers = dask_client.scheduler_info()["workers"]
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
    # topic: Scheduler info shows the registered workers.
    # --- lecture ---
    # Scheduler info shows the registered workers. If this dict were empty,
    # da.compute would have nowhere to run distributed tasks and the lesson
    # would not be testing what we think it is testing.
    # --- facts at this step ---
    #   n_workers = 2
    #   addresses = ['tcp://127.0.0.1:37387', 'tcp://127.0.0.1:45041']
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   full_compare = True
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    assert workers, "expected the LocalCluster to have registered workers"

    adata = _meta_adata(store)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
    # topic: Next we attach layers['counts'] using the gene-block lazy loader.
    # --- lecture ---
    # Next we attach layers['counts'] using the gene-block lazy loader.
    # After this assignment, adata looks like a normal AnnData to scanpy,
    # but the counts values are a dask array whose column chunks map to
    # block_* files.
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   full_compare = True
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
    adata.layers["counts"] = load_geneblock_csc(store, "layers/counts")
    counts = adata.layers["counts"]
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
    # topic: Counts are attached.
    # --- lecture ---
    # Counts are attached. We now assert two structural properties that
    # define success for this lesson's data layout:
    #
    # • the number of gene-axis dask blocks equals the number of on-disk
    #   block_* names
    # • a single gene chunk is narrower than the full gene axis (so we
    #   truly have multiple column chunks)
    #
    # Only after those checks do we run HVG.
    # --- facts at this step ---
    #   shape = (68579, 20387)
    #   numblocks = (1, 11)
    #   chunksize = (68579, 2000)
    #   n_disk_blocks = 11
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   full_compare = True
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
    #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    assert counts.numblocks[1] == len(names)
    assert counts.chunksize[1] != counts.shape[1]
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
    # topic: Structural checks passed.
    # --- lecture ---
    # Structural checks passed. We wrap the HVG call in
    # distributed.get_task_stream so the scheduler records which tasks ran
    # on workers. After HVG returns we assert that the stream is non-empty:
    # proof that work left the client and ran on the cluster.
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   full_compare = True
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
    #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    #   @py_assert0 = None
    #   @py_assert5 = None
    #   @py_assert2 = None
    #   @py_assert3 = None

    with distributed.get_task_stream(dask_client) as ts:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
        # topic: Task-stream recording is now on.
        # --- lecture ---
        # Task-stream recording is now on. Any distributed tasks scheduled
        # during the next highly_variable_genes call should appear in
        # ts.data afterward. Client-only numpy work (for example loess
        # fitting) will not appear there — only scheduler-visible tasks.
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
        #   full_compare = True
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
        #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
        #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        #   @py_assert0 = None
        #   @py_assert5 = None
        #   @py_assert2 = None
        #   @py_assert3 = None
        #   ts = <distributed.diagnostics.task_stream.get_task_stream object at 0x766e15252510>
        result = _run_seurat_v3(adata)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
    # topic: Task-stream context exited, so recording stopped.
    # --- lecture ---
    # Task-stream context exited, so recording stopped. n_tasks > 0 means
    # the distributed scheduler actually executed work for us. That is the
    # difference between "we imported distributed" and "HVG used the
    # cluster".
    # --- facts at this step ---
    #   n_tasks = 341
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   full_compare = True
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
    #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    #   @py_assert0 = None
    #   @py_assert5 = None
    #   @py_assert2 = None
    #   @py_assert3 = None
    #   ts = <distributed.diagnostics.task_stream.get_task_stream object at 0x766e15252510>
    #   result = {'type': 'DataFrame', 'shape': (20387, 6)}

    assert len(ts.data) > 0, "no tasks executed on the distributed cluster"
    assert np.isfinite(result["variances_norm"]).all()
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_assert_parallel_hvg  where=client  pid=2237068  hits_at_site=1
    # topic: The HVG table looks numerically healthy: every variances_norm entry is finite (no NaNs/Infs from divide-by-zero disasters).
    # --- lecture ---
    # The HVG table looks numerically healthy: every variances_norm entry
    # is finite (no NaNs/Infs from divide-by-zero disasters). We also know
    # distributed tasks ran. If full_compare is True, we still owe ourselves
    # an eager reference comparison.
    # --- facts at this step ---
    #   n_hvg = 20387
    #   variances_norm_finite = True
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   full_compare = True
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
    #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    #   @py_assert0 = None
    #   @py_assert5 = None
    #   @py_assert2 = None
    #   @py_assert3 = None
    #   ts = <distributed.diagnostics.task_stream.get_task_stream object at 0x766e15252510>
    #   result = {'type': 'DataFrame', 'shape': (20387, 6)}
    #   @py_assert4 = None
    #   @py_assert7 = None
    #   @py_assert6 = None
    #   @py_assert1 = None
    #   @py_assert9 = None

    if full_compare:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_assert_parallel_hvg  where=compare  pid=2237068  hits_at_site=1
        # topic: ========== COMPARISON CHAPTER ========== We now repeat HVG on an in-memory CSC matrix built from the same gene blocks.
        # --- lecture ---
        # ========== COMPARISON CHAPTER ==========
        #
        # We now repeat HVG on an in-memory CSC matrix built from the same
        # gene blocks. Prefixes switch to [L7 compare] so you can see that
        # this is a second pass, not more parallel worker work for the lazy
        # path.
        #
        # Mentally separate the two runs: the first proved parallelism; this
        # one proves agreement with a simple single-process implementation.
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
        #   full_compare = True
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
        #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
        #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        #   @py_assert0 = None
        #   @py_assert5 = None
        #   @py_assert2 = None
        #   @py_assert3 = None
        #   ts = <distributed.diagnostics.task_stream.get_task_stream object at 0x766e15252510>
        #   result = {'type': 'DataFrame', 'shape': (20387, 6)}
        #   @py_assert4 = None
        #   @py_assert7 = None
        #   @py_assert6 = None
        #   @py_assert1 = None
        #   @py_assert9 = None
        ref = _meta_adata(store)
        ref.layers["counts"] = load_geneblock_csc_memory(store, "layers/counts")
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_assert_parallel_hvg  where=compare  pid=2237068  hits_at_site=1
        # topic: Reference counts are a scipy sparse matrix in this process.
        # --- lecture ---
        # Reference counts are a scipy sparse matrix in this process. When
        # seurat_v3 runs now, clip_square_sum will take the CSBase/numba
        # path instead of the dask map_blocks path. Same formulas, different
        # execution engine.
        # --- facts at this step ---
        #   type = 'csc_matrix'
        #   shape = (68579, 20387)
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
        #   full_compare = True
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
        #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
        #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        #   @py_assert0 = None
        #   @py_assert5 = None
        #   @py_assert2 = None
        #   @py_assert3 = None
        #   ts = <distributed.diagnostics.task_stream.get_task_stream object at 0x766e15252510>
        #   result = {'type': 'DataFrame', 'shape': (20387, 6)}
        #   @py_assert4 = None
        #   @py_assert7 = None
        #   @py_assert6 = None
        #   @py_assert1 = None
        #   @py_assert9 = None
        #   ref = {'type': 'AnnData', 'shape': (68579, 20387)}
        reference = _run_seurat_v3(ref)
        assert_hvg_close(result, reference)
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_assert_parallel_hvg  where=compare  pid=2237068  hits_at_site=1
        # topic: Comparison finished successfully.
        # --- lecture ---
        # Comparison finished successfully. The lazy LocalCluster answer and
        # the eager in-memory answer match within tolerance. That is the
        # full Lesson 7 punchline: out-of-core gene-block parallelism without
        # changing the biological result.
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
        #   full_compare = True
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
        #   adata = {'type': 'AnnData', 'shape': (68579, 20387)}
        #   counts = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        #   @py_assert0 = None
        #   @py_assert5 = None
        #   @py_assert2 = None
        #   @py_assert3 = None
        #   ts = <distributed.diagnostics.task_stream.get_task_stream object at 0x766e15252510>
        #   result = {'type': 'DataFrame', 'shape': (20387, 6)}
        #   @py_assert4 = None
        #   @py_assert7 = None
        #   @py_assert6 = None
        #   @py_assert1 = None
        #   @py_assert9 = None
        #   ref = {'type': 'AnnData', 'shape': (68579, 20387)}
        #   reference = {'type': 'DataFrame', 'shape': (20387, 6)}


@pytest.mark.slow
def test_lesson7a_pbmc3k_localcluster(dask_client):
    """7A: full pbmc3k gene-block store + LocalCluster seurat_v3 vs eager hstack."""
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # Lesson 7A uses the classic pbmc3k dataset: a few thousand cells and,
    # after the fixture's gene filtering, on the order of ~16k genes stored
    # as multiple CSC gene-block subgroups (each block is typically 2000
    # genes wide, e.g. shape (2700, 2000) per strip). The store is built
    # once and reused. This is the faster of the two Lesson 7 exams — ideal
    # while you read the tutor narration.
    store = build_geneblock_fixtures.build("pbmc3k", force=False)
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # Store path is ready (either freshly built or reused from data/). We
    # hand it to the shared assertion helper with full_compare=True.
    _assert_parallel_hvg(store, dask_client, full_compare=True)
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # Lesson 7A passed end-to-end. You have exercised gene-block lazy load,
    # LocalCluster seurat_v3, task-stream evidence, and eager agreement.


def test_lesson7b_pbmc68k_localcluster(dask_client):
    """7B: 10x 68k gene-block store + LocalCluster seurat_v3."""
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=test_lesson7b_pbmc68k_localcluster  where=client  pid=2237068  hits_at_site=1
    # topic: Lesson 7B repeats the same logic on a much larger matrix.
    # --- lecture ---
    # Lesson 7B repeats the same logic on a much larger matrix. On this
    # machine the gene-block fixture is about ~68k cells × ~20k genes in
    # roughly eleven on-disk CSC strips — same algorithm as 7A, bigger
    # parcels for each worker. If downloads are disabled via
    # HVG_LESSON7_SKIP_DOWNLOAD=1 and the store is missing, this test would
    # not get this far.
    #
    # Lecture analysis of the live Lesson 7B state:
    #   • This is the 7B entry point — the lecture-comment base dataset.
    #   • dask_client already shows processes=2, threads=2 (two 1-thread workers).
    #   • After build/load you should see store=.../pbmc68k_geneblocks.zarr and
    #     counts shape (68579, 20387) with numblocks (1, 11).
    #
    # --- locals / object fields at the call site ---
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    store = build_geneblock_fixtures.build("pbmc68k", force=False)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=test_lesson7b_pbmc68k_localcluster  where=client  pid=2237068  hits_at_site=1
    # topic: pbmc68k gene-block store is ready.
    # --- lecture ---
    # pbmc68k gene-block store is ready. Same exam as 7A, bigger numbers.
    # --- facts at this step ---
    #   store = '/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr'
    # --- locals / object fields at the call site ---
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    _assert_parallel_hvg(store, dask_client, full_compare=True)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=test_lesson7b_pbmc68k_localcluster  where=client  pid=2237068  hits_at_site=1
    # topic: Lesson 7B passed.
    # --- lecture ---
    # Lesson 7B passed. At this scale, the benefit of not holding the full
    # dense matrix in one worker becomes much more intuitive: each gene
    # strip can be read and reduced independently.
    # --- locals / object fields at the call site ---
    #   dask_client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
