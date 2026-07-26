from __future__ import annotations

import os
from functools import singledispatch
from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from fast_array_utils import stats

from ... import logging as logg
from ..._compat import CSBase, CSCBase, CSRBase, DaskArray, warn
from ..._utils import (
    check_nonnegative_integers,
    raise_if_dask_feature_axis_chunked,
)
from ...get import _get_obs_rep, aggregate
from .._distributed import materialize_as_ndarray
if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


def _raise_if_unsupported_dask_chunking(data) -> None:
    """Reject dask chunkings that seurat_v3 cannot handle.

    Row-chunked (or unchunked-feature) dask arrays are supported, as is
    column-chunked ``csc``-in-dask (the observation axis whole, feature axis
    chunked). Everything else that chunks the feature axis - dense or ``csr``
    feature-chunked, or ``csc`` chunked on both axes - is rejected with the
    standard message.
    """
    # Non-dask inputs need no chunking checks.
    if not isinstance(data, DaskArray):
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_raise_if_unsupported_dask_chunking  where=scanpy  pid=2237068  hits_at_site=1
        # topic: Before Seurat-v3 highly-variable-gene math can begin, we ask a simple gatekeeping question: is this expression matrix a Dask array at all?
        # --- lecture ---
        # Before Seurat-v3 highly-variable-gene math can begin, we ask a simple
        # gatekeeping question: is this expression matrix a Dask array at all?
        #
        # Right now the answer is no. That means the matrix already lives (or
        # will be handled) as an ordinary in-memory NumPy or SciPy object in
        # this process. There is no lazy graph of chunks, no worker pool that
        # must receive tasks, and therefore no "chunking layout" that could be
        # illegal for this algorithm.
        #
        # Think of Dask as a promise to compute later in pieces. If you never
        # made that promise — if you just have a normal array — then the
        # special rules about which pieces are allowed do not apply. We accept
        # the input and move on without calling the feature-axis chunking
        # checker.
        # --- facts at this step ---
        #   type = <class 'scipy.sparse._csc.csc_matrix'>
        # --- locals / object fields at the call site ---
        #   data = {'type': 'csc_matrix', 'shape': (68579, 20387), 'dtype': 'float32', 'nnz': 37323295}
        return
    # Feature axis unchunked (one column chunk spanning all genes).
    if data.chunksize[1] == data.shape[1]:
        # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
        # Alternate branch unused by test_lesson7b; no live 7B values here.
        # --- lecture ---
        # This input is a Dask array, so in principle it could be split along
        # cells, along genes, or both. The feature axis is the gene axis —
        # columns in the usual cells×genes layout.
        #
        # Here, the gene-axis chunk size equals the full number of genes. In
        # plain language: every Dask block still contains every gene. The only
        # possible splitting is along cells (rows). That is the classic
        # "row-chunked" layout: each task sees a strip of cells and all genes.
        #
        # Seurat v3 can handle that layout, because later steps that reduce
        # over cells can sum partial contributions across those row blocks.
        # So we accept this chunking and do not raise.
        return
    # Lesson-7 path: CSC meta + full-height column chunks.
    if isinstance(data._meta, CSCBase) and data.chunksize[0] == data.shape[0]:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_raise_if_unsupported_dask_chunking  where=scanpy  pid=2237068  hits_at_site=1
        # topic: This is the Lesson-7 friendly layout: a Dask array whose prototype (``_meta``) is a CSC sparse matrix, and whose observation-axis chunk size equals the full number of cells.
        # --- lecture ---
        # This is the Lesson-7 friendly layout: a Dask array whose prototype
        # (``_meta``) is a CSC sparse matrix, and whose observation-axis chunk
        # size equals the full number of cells. That means each gene-column
        # chunk is "full height" — every cell for a contiguous block of genes.
        #
        # Why CSC? Compressed Sparse Column stores data column-by-column, so
        # grabbing a contiguous set of gene columns is natural. Combined with
        # "all cells in every block," each worker can finish the clipped
        # sum / sum-of-squares for its gene block without needing another
        # gene's data from a neighbor chunk.
        #
        # Analogy: imagine a spreadsheet of cells (rows) × genes (columns).
        # We tear the sheet into vertical strips. Each strip has every row but
        # only some columns. That is exactly what column-chunked, full-height
        # CSC-in-Dask means — and Seurat v3 accepts it.
        # --- facts at this step ---
        #   shape = (68579, 20387)
        #   chunksize = (68579, 2000)
        #   numblocks = (1, 11)
        #   meta_type = 'csc_matrix'
        # --- locals / object fields at the call site ---
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        return
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # We reached the rejection path. The array is Dask, the gene axis is
    # chunked (so blocks do not each contain every gene), and it is not the
    # special full-height CSC column-chunked case we just described.
    #
    # Examples of layouts that land here include dense or CSR arrays chunked
    # along genes, or CSC arrays that are also chunked along cells. Those
    # layouts make the Seurat-v3 reductions awkward or incorrect with the
    # current implementation: a gene's clipped statistics might be split
    # across blocks in a way this code does not reassemble.
    #
    # Next we call the shared helper that raises a clear error about
    # unsupported feature-axis chunking, so the user can rechunk rather than
    # get a silent wrong answer.
    raise_if_dask_feature_axis_chunked(data)


@singledispatch
def clip_square_sum(
    data_batch: np.ndarray, clip_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[DaskArray, DaskArray]:
    """Clip data_batch by clip_val.

    Parameters
    ----------
    data_batch
        The data to be clipped
    clip_val
        Clip by these values (must be broadcastable to the input data)

    Returns
    -------
        The clipeed data
    """
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # ``clip_square_sum`` is a singledispatch function: Python picks an
    # implementation based on the type of ``data_batch``. This is the default
    # path for a dense NumPy ``ndarray``.
    #
    # Scientifically, Seurat's VST step does not want a few huge outlier
    # counts to dominate a gene's variance. So for each gene we have a clip
    # threshold (``clip_val``). Any count larger than that threshold is pulled
    # down to the threshold — like a speed limit for expression values —
    # before we accumulate the sum and the sum of squares.
    #
    # On this dense path we copy to float64, broadcast the per-gene clip
    # values across cells, apply ``np.putmask``, then sum down the cell axis.
    # The return value is two vectors (one number per gene): sum of squares
    # of clipped counts, and sum of clipped counts.
    batch_counts = data_batch.astype(np.float64).copy()
    clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
    np.putmask(
        batch_counts,
        batch_counts > clip_val_broad,
        clip_val_broad,
    )

    squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
    batch_counts_sum = batch_counts.sum(axis=0)
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # The dense clip-and-reduce finished. We now hold two length-``n_genes``
    # vectors.
    #
    # ``squared_batch_counts_sum`` is Σ (clipped count)² over cells, for each
    # gene. ``batch_counts_sum`` is Σ (clipped count) over cells. Together
    # with the gene means and the regularized standard deviations computed
    # earlier on the client, these two summaries are enough to rebuild the
    # normalized variance without shipping the full cells×genes block again.
    #
    # That is the whole point of this helper: compress a tall matrix into two
    # short vectors that still carry the information the VST formula needs.
    return squared_batch_counts_sum, batch_counts_sum


@clip_square_sum.register(DaskArray)
def _(data_batch: DaskArray, clip_val: np.ndarray) -> tuple[DaskArray, DaskArray]:
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_  where=scanpy  pid=2237068  hits_at_site=1
    # topic: ``clip_square_sum`` was called on a Dask array.
    # --- lecture ---
    # ``clip_square_sum`` was called on a Dask array. We are still on the
    # scanpy client process right now; we have not necessarily loaded all
    # counts into RAM. Instead we will build (or dispatch to) a lazy
    # computation graph whose tasks know how to clip and reduce each block.
    #
    # Two layouts matter. If genes are split across blocks (feature-chunked),
    # we take the Lesson-7 helper that maps one task per gene block. If genes
    # are not split — only cells are chunked — we map a function over row
    # blocks and sum the partial sums. Either way, the public return type is
    # still "two arrays of per-gene statistics," which may themselves still
    # be lazy Dask arrays until someone calls ``compute``.
    # --- facts at this step ---
    #   shape = (68579, 20387)
    #   chunksize = (68579, 2000)
    #   numblocks = (1, 11)
    #   clip_val_shape = (20387,)
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    # Column-chunked: each block is final for its genes (Lesson 7 path).
    if data_batch.chunksize[1] != data_batch.shape[1]:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_  where=scanpy  pid=2237068  hits_at_site=1
        # topic: The gene-axis chunk size is smaller than the total number of genes, so this Dask array is feature-chunked: different blocks own different gene ranges.
        # --- lecture ---
        # The gene-axis chunk size is smaller than the total number of genes,
        # so this Dask array is feature-chunked: different blocks own different
        # gene ranges.
        #
        # For that layout we do not want to sum across gene blocks (that would
        # mix unrelated genes). Each block's clipped sums are already final for
        # its own genes. We hand off to ``_clip_square_sum_feature_chunked``,
        # which uses ``map_blocks`` so each gene strip can be processed —
        # often on a distributed worker — independently.
        # --- facts at this step ---
        #   gene_chunksize = 2000
        #   n_genes = 20387
        # --- locals / object fields at the call site ---
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        return _clip_square_sum_feature_chunked(data_batch, clip_val)

    # Row-chunked: sum clipped contributions across observation blocks.
    n_blocks = data_batch.blocks.size
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # This Dask array is row-chunked: every block still has all genes, but
    # only a subset of cells. Clipped sums from one row block are only part
    # of the story; we must add the contributions from every cell strip.
    #
    # So we ``map_blocks`` a small wrapper that runs the ordinary
    # ``clip_square_sum`` on each dense/sparse block, stacks the two summary
    # vectors, and then ``.sum(axis=0)`` across blocks. Addition is the right
    # merge because sum and sum-of-squares both accumulate linearly over cells.

    def sum_and_sum_squares_clipped_from_block(block):
        return np.vstack(clip_square_sum(block, clip_val))[None, ...]

    squared_batch_counts_sum, batch_counts_sum = data_batch.map_blocks(
        sum_and_sum_squares_clipped_from_block,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (2,), (data_batch.shape[1],)),
        meta=np.array([]),
        dtype=np.float64,
    ).sum(axis=0)
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # The row-chunked graph is assembled. What we hold are still lazy Dask
    # arrays (unless something already forced a compute). Conceptually they
    # already mean "total clipped sum-of-squares per gene" and "total clipped
    # sum per gene," but the arithmetic may not have run yet.
    #
    # Downstream Seurat-v3 code can keep building formulas with these lazy
    # objects; a later ``da.compute`` (or materialization) is when workers
    # actually touch the count data.
    return squared_batch_counts_sum, batch_counts_sum


def _clip_square_sum_feature_chunked(
    data_batch: DaskArray, clip_val: np.ndarray
) -> tuple[DaskArray, DaskArray]:
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_clip_square_sum_feature_chunked  where=scanpy  pid=2237068  hits_at_site=1
    # topic: Entering the feature-chunked clip helper.
    # --- lecture ---
    # Entering the feature-chunked clip helper. Preconditions we care about:
    # the observation axis must be a single chunk (full height), so each gene
    # block already contains every cell for those genes. If cells were also
    # split, a single gene's clipped sum would be scattered across several
    # blocks and we would need a second reduction stage we do not implement
    # here.
    #
    # Next we define ``per_block``, a function that will run once per gene
    # chunk — possibly in another process on a Dask worker — and we wire it
    # into ``map_blocks``. Building that graph is still lazy; no worker has
    # to run until compute time.
    # --- facts at this step ---
    #   shape = (68579, 20387)
    #   numblocks = (1, 11)
    #   chunks = ((68579,), (2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 387))
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    if data_batch.numblocks[0] != 1:
        msg = "clip_square_sum requires the observation axis to be unchunked for feature-chunked dask arrays."
        raise ValueError(msg)

    def per_block(block, block_info: dict | None = None) -> np.ndarray:
        # Worker task: one gene-block of CSC counts.
        col_subset = slice(*block_info[0]["array-location"][1])
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=per_block  where=worker  pid=2237353  hits_at_site=11
        # topic: A Dask worker just received one gene-block task.
        # --- lecture ---
        # A Dask worker just received one gene-block task. This process is not
        # the original scanpy client; it is a worker process that was given a
        # sparse (or dense) block covering all cells but only a slice of genes.
        #
        # ``block_info`` tells us where this block sits in the global array.
        # From the gene-axis ``array-location`` we build ``col_subset``, a
        # Python ``slice`` naming which columns of the full matrix this block
        # owns. We use that same slice to select the matching entries of
        # ``clip_val``, because clip thresholds are per-gene and must line up
        # with the columns inside this block.
        #
        # Analogy: the full matrix is a long bookshelf of gene volumes. This
        # worker was handed volumes 400–500 only, plus the clip-speed-limits
        # that belong to exactly those volumes.
        #
        # Lecture analysis of the live Lesson 7B state:
        #   • Worker PID 2237353 ≠ pytest/client — this is a LocalCluster worker.
        #   • block is CSC (68579, 2000), float32, nnz=4_879_122 — all ~68k cells,
        #     one 2000-gene strip (much denser nonzeros than the smaller 7A strips).
        #   • Global matrix in block_info: shape (68579, 20387), 11 gene chunks.
        #   • This task owns genes [18000, 20000) → col_subset / chunk-location (0, 9).
        #   • clip_val length 20387; clip_slice_len 2000 matches this strip.
        #   • hits_at_site=11 ≈ one lecture hit per gene chunk on this worker path.
        #
        # --- facts at this step ---
        #   pid = 2237353
        #   col_subset = slice(18000, 20000, None)
        #   block_shape = (68579, 2000)
        #   block_type = 'csc_matrix'
        #   clip_slice_len = 2000
        # --- locals / object fields at the call site ---
        #   block = {'type': 'csc_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
        #   block_info = {0: {'shape': (68579, 20387), 'num-chunks': (1, 11), 'array-location': [(0, 68579), (18000, 20000)], 'chunk-location': (0, 9)}, ...}
        #   col_subset = slice(18000, 20000, None)
        #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        # Clip + sum on this block (often CSBase → numba path).
        squared_sum, total = clip_square_sum(block, clip_val[col_subset])
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=per_block  where=worker  pid=2237353  hits_at_site=11
        # topic: ``clip_square_sum`` returned for this gene block.
        # --- lecture ---
        # ``clip_square_sum`` returned for this gene block. For each gene in
        # the block we now have the sum of clipped counts and the sum of
        # squared clipped counts, reduced over all cells that live in this
        # block (which, on the Lesson-7 path, is every cell).
        #
        # Those two vectors are ordinary NumPy arrays in this worker's memory.
        # They are much smaller than the original sparse block: we went from
        # a cells×genes_chunk matrix down to two length-genes_chunk vectors.
        # That compression is why out-of-core / distributed HVG is feasible —
        # we ship summaries, not whole count matrices, back into the larger
        # graph.
        # --- facts at this step ---
        #   pid = 2237353
        #   squared_sum_len = 2000
        #   total_len = 2000
        # --- locals / object fields at the call site ---
        #   block = {'type': 'csc_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
        #   block_info = {0: {'shape': (68579, 20387), 'num-chunks': (1, 11), 'array-location': [(0, 68579), (18000, 20000)], 'chunk-location': (0, 9)}, None: {'shape': (2, 20387), '...
        #   col_subset = slice(18000, 20000, None)
        #   squared_sum = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
        #   total = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
        #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        stacked = np.vstack([np.asarray(squared_sum), np.asarray(total)])
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=per_block  where=worker  pid=2237353  hits_at_site=11
        # topic: For ``map_blocks`` we need a single ndarray return value whose shape matches the declared chunks ``(2, genes_in_this_block)``.
        # --- lecture ---
        # For ``map_blocks`` we need a single ndarray return value whose shape
        # matches the declared chunks ``(2, genes_in_this_block)``. So we stack
        # the two summary vectors as rows: row 0 is sum of squares, row 1 is
        # the plain sum.
        #
        # After all blocks finish, the client-side code will index
        # ``combined[0]`` and ``combined[1]`` to peel those rows apart again
        # into two lazy vectors aligned with the full gene axis. Stacking is
        # only a packaging convention between worker and graph.
        # --- facts at this step ---
        #   pid = 2237353
        #   stacked_shape = (2, 2000)
        # --- locals / object fields at the call site ---
        #   block = {'type': 'csc_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
        #   block_info = {0: {'shape': (68579, 20387), 'num-chunks': (1, 11), 'array-location': [(0, 68579), (18000, 20000)], 'chunk-location': (0, 9)}, None: {'shape': (2, 20387), '...
        #   col_subset = slice(18000, 20000, None)
        #   squared_sum = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
        #   total = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
        #   stacked = {'type': 'ndarray', 'shape': (2, 2000), 'dtype': 'float64'}
        #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        return stacked

    # Build lazy graph only — workers run per_block later at da.compute.
    combined = data_batch.map_blocks(
        per_block,
        chunks=((2,), data_batch.chunks[1]),
        meta=np.array([], dtype=np.float64),
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_clip_square_sum_feature_chunked  where=scanpy  pid=2237068  hits_at_site=1
    # topic: Back on the client: ``map_blocks`` has registered the ``per_block`` function on every gene chunk, but those tasks have not necessarily run.
    # --- lecture ---
    # Back on the client: ``map_blocks`` has registered the ``per_block``
    # function on every gene chunk, but those tasks have not necessarily run.
    # ``combined`` is a lazy Dask array with two rows and one column per gene
    # (chunked the same way as the input gene axis).
    #
    # This is the moment students often confuse "building the recipe" with
    # "cooking the meal." We only wrote the recipe. The meal — workers loading
    # CSC blocks, clipping, summing — happens when something later forces
    # compute (for example ``da.compute`` on the normalized variances).
    # --- facts at this step ---
    #   combined_shape = (2, 20387)
    #   combined_numblocks = (1, 11)
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   combined = {'type': 'Array', 'shape': (2, 20387), 'dtype': 'float64', 'numblocks': (1, 11), 'chunksize': (2, 2000), 'meta_type': 'ndarray', 'meta_format': None}
    squared = combined[0]
    totals = combined[1]
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_clip_square_sum_feature_chunked  where=scanpy  pid=2237068  hits_at_site=1
    # topic: We split the stacked lazy array into two views: ``squared`` is the first row (per-gene sum of squares of clipped counts) and ``totals`` is the second row (per-gene sum of clipped counts).
    # --- lecture ---
    # We split the stacked lazy array into two views: ``squared`` is the first
    # row (per-gene sum of squares of clipped counts) and ``totals`` is the
    # second row (per-gene sum of clipped counts).
    #
    # Indexing a Dask array like this does not copy data; it adds thin layers
    # on the graph. Callers of ``clip_square_sum`` can treat the two return
    # values just like the NumPy path's two vectors, whether or not the
    # underlying tasks have executed yet.
    # --- facts at this step ---
    #   squared_shape = (20387,)
    #   totals_shape = (20387,)
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   clip_val = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   combined = {'type': 'Array', 'shape': (2, 20387), 'dtype': 'float64', 'numblocks': (1, 11), 'chunksize': (2, 2000), 'meta_type': 'ndarray', 'meta_format': None}
    #   squared = {'type': 'Array', 'shape': (20387,), 'dtype': 'float64', 'numblocks': (11,), 'chunksize': (2000,), 'meta_type': 'ndarray', 'meta_format': None}
    #   totals = {'type': 'Array', 'shape': (20387,), 'dtype': 'float64', 'numblocks': (11,), 'chunksize': (2000,), 'meta_type': 'ndarray', 'meta_format': None}
    return squared, totals


@clip_square_sum.register(CSBase)
def _(data_batch: CSBase, clip_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_  where=scanpy  pid=2237353  hits_at_site=12
    # topic: This ``clip_square_sum`` overload handles SciPy-style sparse matrices (CSR or CSC) via the ``CSBase`` ABC.
    # --- lecture ---
    # This ``clip_square_sum`` overload handles SciPy-style sparse matrices
    # (CSR or CSC) via the ``CSBase`` ABC. Sparse matrices store only nonzero
    # entries, which is how single-cell count matrices usually look: most
    # gene–cell pairs are zero.
    #
    # We still need the same two per-gene summaries (sum and sum of squares
    # after clipping). Walking a dense cells×genes array would waste time and
    # memory on all those zeros. Instead we will convert to CSR if needed and
    # call a Numba kernel that loops only over the nonzero values.
    # --- facts at this step ---
    #   shape = (68579, 2000)
    #   format = 'csc'
    #   nnz = 4879122
    #   clip_val_shape = (2000,)
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'csc_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
    #   clip_val = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
    batch_counts = data_batch if isinstance(data_batch, CSRBase) else data_batch.tocsr()
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_  where=scanpy  pid=2237353  hits_at_site=12
    # topic: The Numba kernel below expects CSR-style ``indices`` and ``data`` arrays: for each stored nonzero, ``indices[i]`` is the gene (column) index and ``data[i]`` is the count.
    # --- lecture ---
    # The Numba kernel below expects CSR-style ``indices`` and ``data`` arrays:
    # for each stored nonzero, ``indices[i]`` is the gene (column) index and
    # ``data[i]`` is the count. CSC is column-oriented; CSR is row-oriented but
    # still carries explicit column indices for every nonzero, which is what
    # our loop uses to bucket values into per-gene accumulators.
    #
    # If the input was already CSR we keep it; if it was CSC we convert once.
    # Either way, after this step ``batch_counts`` is ready for the compiled
    # clip-and-accumulate loop.
    # --- facts at this step ---
    #   format = 'csr'
    #   nnz = 4879122
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'csc_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
    #   clip_val = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
    #   batch_counts = {'type': 'csr_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}

    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_  where=scanpy  pid=2237353  hits_at_site=12
    # topic: We are about to call ``_sum_and_sum_squares_clipped``, a Numba ``@njit`` function.
    # --- lecture ---
    # We are about to call ``_sum_and_sum_squares_clipped``, a Numba
    # ``@njit`` function. Numba compiles a restricted subset of Python to
    # machine code. That is great for a tight loop over millions of sparse
    # nonzeros, but it means we cannot run ordinary Python lecture helpers
    # from inside the jitted function body.
    #
    # So the teaching notes live here, immediately before the call, and
    # again immediately after it returns. Mentally inline what the kernel
    # does: for each nonzero count, clip it to that gene's ``clip_val``, then
    # add the clipped value and its square into the gene's two accumulators.
    # Zeros never appear in the sparse structure, so they correctly contribute
    # nothing.
    # --- facts at this step ---
    #   n_cols = 2000
    #   nnz = 4879122
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'csc_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
    #   clip_val = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
    #   batch_counts = {'type': 'csr_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
    out = _sum_and_sum_squares_clipped(
        batch_counts.indices,
        batch_counts.data,
        n_cols=batch_counts.shape[1],
        clip_val=clip_val,
        nnz=batch_counts.nnz,
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_  where=scanpy  pid=2237353  hits_at_site=12
    # topic: The Numba kernel returned.
    # --- lecture ---
    # The Numba kernel returned. ``out[0]`` is the per-gene sum of squares of
    # clipped nonzero contributions; ``out[1]`` is the per-gene sum of clipped
    # values. Genes that were all zero in this block stay at zero in both
    # vectors, which is correct.
    #
    # From the caller's point of view this is identical to the dense path's
    # result type — two NumPy vectors — just computed without materializing a
    # dense matrix. If we were inside a Dask worker's ``per_block``, these
    # vectors will next be stacked and returned to the graph.
    # --- facts at this step ---
    #   squared_len = 2000
    #   sum_len = 2000
    # --- locals / object fields at the call site ---
    #   data_batch = {'type': 'csc_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
    #   clip_val = {'type': 'ndarray', 'shape': (2000,), 'dtype': 'float64'}
    #   batch_counts = {'type': 'csr_matrix', 'shape': (68579, 2000), 'dtype': 'float32', 'nnz': 4879122}
    #   out = (array([ 458.,   96., 5594., ..., 6406., 4645., 1775.], shape=(2000,)), array([ 410.,   90., 4320., ..., 5202., 3811., 1527.], shape=(2000,)))
    return out


# parallel=False needed for accuracy
@numba.njit(cache=True, parallel=False)  # noqa: TID251
def _sum_and_sum_squares_clipped(
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    *,
    n_cols: int,
    clip_val: NDArray[np.float64],
    nnz: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    squared_batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    for i in numba.prange(nnz):
        idx = indices[i]
        element = min(np.float64(data[i]), clip_val[idx])
        squared_batch_counts_sum[idx] += element**2
        batch_counts_sum[idx] += element

    return squared_batch_counts_sum, batch_counts_sum


def _highly_variable_genes_seurat_v3(  # noqa: PLR0912, PLR0915
    adata: AnnData,
    *,
    flavor: Literal["seurat_v3", "seurat_v3_paper"] = "seurat_v3",
    layer: str | None = None,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    check_values: bool = True,
    span: float = 0.3,
    subset: bool = False,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """See `highly_variable_genes`.

    For further implementation details see https://www.overleaf.com/read/ckptrbgzzzpg

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields:

    highly_variable : :class:`bool`
        boolean indicator of highly-variable genes.
    **means**
        means per gene.
    **variances**
        variance per gene.
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches.
    highly_variable_rank : :class:`float`
        Rank of the gene according to normalized variance, median rank in the case of multiple batches.
    highly_variable_nbatches : :class:`int`
        If batch_key is given, this denotes in how many batches genes are detected as HVG.

    """
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Welcome to the Seurat v3 / Seurat v3 paper implementation of highly variable gene (HVG) selection.
    # --- lecture ---
    # Welcome to the Seurat v3 / Seurat v3 paper implementation of highly
    # variable gene (HVG) selection. The public ``highly_variable_genes``
    # function has already decided that your ``flavor`` is one of these two
    # VST-style methods and has delegated here.
    #
    # Unlike the older dispersion flavors, Seurat v3 expects raw integer
    # counts, fits a smooth mean–variance trend with LOESS, clips counts
    # using a regularized standard deviation, and then ranks genes by a
    # normalized variance. Batches, if requested, are handled by repeating
    # that logic per batch and combining ranks.
    #
    # The keyword arguments you see in the facts appendix are the knobs for
    # that whole story: which layer holds counts, how many top genes to keep,
    # whether to write into ``adata.var`` or return a DataFrame, and so on.
    #
    # The lecture-comment base (Lesson 7B) typically shows
    # ``adata_shape`` around (68579, 20387). Lesson 7A is the same pipeline
    # on a smaller matrix (~2700 × ~16k). The algorithms below do not change
    # with scale — only how expensive each gene-strip task becomes.
    #
    # Lecture analysis of the live Lesson 7B state (first / lazy call):
    #   • adata is 68579 obs × 20387 vars with layers['counts'] (dask CSC strips).
    #   • flavor='seurat_v3', layer='counts', batch_key=None (single batch).
    #   • n_top_genes=20387 means "score every gene" for the test comparison.
    #   • inplace=False → return a DataFrame; subset=False → keep all genes.
    #   • hits_at_site=2 because full_compare runs seurat_v3 again on the eager
    #     hstack reference with the same kwargs.
    #
    # --- facts at this step ---
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   adata_shape = (68579, 20387)
    # --- locals / object fields at the call site ---
    #   adata = AnnData n_obs × n_vars = 68579 × 20387; obs: batch; var: gene_ids, n_cells; layers: counts
    #   flavor/layer/n_top_genes/batch_key/check_values/span/subset/inplace as above
    try:
        from skmisc.loess import loess
    except ImportError as e:
        e.add_note("Please install `scikit-misc` and try again.")
        raise
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: The LOESS smoother from ``skmisc.loess`` imported successfully.
    # --- lecture ---
    # The LOESS smoother from ``skmisc.loess`` imported successfully. LOESS
    # is a locally weighted polynomial regression: around each point on the
    # log-mean axis it fits a small curve using nearby genes, controlled by
    # the ``span`` fraction.
    #
    # We need that smoother because raw variances grow with mean expression
    # in a complicated way. Seurat v3 estimates "what variance would we
    # expect at this mean?" with LOESS, then asks which genes are unusually
    # variable relative to that expectation. Without ``scikit-misc``, this
    # flavor cannot run — that is why the import is mandatory here.
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False

    # Per-gene results table (index = gene names).
    df = pd.DataFrame(index=adata.var_names)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: We create an empty results table whose row index is the gene names from ``adata.var_names``.
    # --- lecture ---
    # We create an empty results table whose row index is the gene names from
    # ``adata.var_names``. Every column we add later — means, variances,
    # normalized variances, ranks, HVG flags — will be aligned to those genes.
    #
    # Think of this DataFrame as the report card that the algorithm fills in
    # gene by gene. It is not the expression matrix; it is one row per gene
    # and eventually a handful of summary columns.
    # --- facts at this step ---
    #   n_genes = 20387
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 0), 'columns': []}

    # Expression matrix from X or the requested layer (may be dask).
    data = _get_obs_rep(adata, layer=layer)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: ``_get_obs_rep`` fetches the expression matrix the user asked for: ``adata.X`` when ``layer`` is None, otherwise ``adata.layers[layer]``.
    # --- lecture ---
    # ``_get_obs_rep`` fetches the expression matrix the user asked for:
    # ``adata.X`` when ``layer`` is None, otherwise ``adata.layers[layer]``.
    # For Lesson 7 that is often a column-chunked CSC Dask array backed by
    # Zarr, but it could also be an in-memory NumPy or SciPy matrix.
    #
    # From this point on, almost all numeric work talks to ``data``, not to
    # the AnnData wrapper. AnnData still matters for gene names, batch labels
    # in ``.obs``, and for writing results back into ``.var`` at the end.
    # --- facts at this step ---
    #   layer = 'counts'
    #   data_type = 'Array'
    #   data_shape = (68579, 20387)
    #   is_dask = True
    #   numblocks = (1, 11)
    #   chunksize = (68579, 2000)
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 0), 'columns': []}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...

    _raise_if_unsupported_dask_chunking(data)

    if check_values and not check_nonnegative_integers(data):
        msg = f"`{flavor=!r}` expects raw count data, but non-integers were found."
        warn(msg, UserWarning)
        # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
        # Alternate branch unused by test_lesson7b; no live 7B values here.
        # --- lecture ---
        # Value checking was enabled, and the matrix failed the "nonnegative
        # integers" test. Seurat v3's VST was derived for raw UMI/count data.
        # Log-normalized floats, centered values, or negative entries are the
        # wrong input scale for this flavor.
        #
        # We emit a UserWarning rather than hard-failing, because some users
        # knowingly proceed. Scientifically, though, you should treat this as
        # a red flag: the LOESS mean–variance fit and the clipping thresholds
        # assume count-like magnitudes.
    else:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: Either value checking was turned off, or the matrix looks like nonnegative integers (as far as the checker can tell).
        # --- lecture ---
        # Either value checking was turned off, or the matrix looks like
        # nonnegative integers (as far as the checker can tell). We continue
        # under the modeling assumption that these are raw counts suitable for
        # Seurat v3 VST.
        #
        # Remember: passing the check does not prove biology is perfect; it
        # only means the dtype/value pattern is consistent with counts. The
        # algorithm will now estimate per-gene means and variances.
        # --- facts at this step ---
        #   check_values = True
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 0), 'columns': []}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        pass

    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Next we compute the per-gene mean and variance along axis 0 (down the cells).
    # --- lecture ---
    # Next we compute the per-gene mean and variance along axis 0 (down the
    # cells). For an in-memory matrix this is a local reduction. For a Dask
    # array, ``stats.mean_var`` builds tasks that may run on workers and will
    # need to touch the actual count chunks.
    #
    # These global mean/variance columns become the baseline gene statistics
    # stored in the results table. When batches are present we will also
    # compute batch-wise means and variances; when not, these globals are
    # reused as the single logical batch.
    # --- facts at this step ---
    #   axis = 0
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 0), 'columns': []}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    df["means"], df["variances"] = stats.mean_var(data, axis=0, correction=1)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Mean and variance per gene are now materialized in the results DataFrame.
    # --- lecture ---
    # Mean and variance per gene are now materialized in the results
    # DataFrame. Low means are genes rarely detected; high variances may be
    # truly interesting biology or may simply be genes with high mean
    # expression (because variance scales with mean in count data).
    #
    # That mean–variance coupling is exactly why the later LOESS step exists:
    # we need a fair, mean-aware notion of "more variable than expected"
    # rather than ranking raw variance alone.
    # --- facts at this step ---
    #   mean_min = 1.4581723268055818e-05
    #   mean_max = 40.29369048834191
    #   var_min = 1.4581723268055818e-05
    #   var_max = 210.56994633664326
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...

    batch_info = (
        pd.Categorical(np.zeros(adata.shape[0], dtype=int))
        if batch_key is None
        else adata.obs[batch_key]
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Batch labels decide whether we run one VST pass or several.
    # --- lecture ---
    # Batch labels decide whether we run one VST pass or several. If the user
    # did not pass ``batch_key``, we invent a single batch of all zeros — one
    # logical group containing every cell. If they did pass a key, we read
    # that column from ``adata.obs`` (typically a categorical like sample or
    # donor).
    #
    # Multi-batch mode is how Seurat-style integration feature selection
    # avoids picking genes that are "variable" only because of batch effects:
    # a gene should look highly variable in many batches to rise to the top.
    # --- facts at this step ---
    #   batch_key = None
    #   n_batches = 1
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'Categorical', 'shape': (68579,), 'dtype': 'category'}
    norm_gene_vars = []

    adata_agg = AnnData(
        X=data,
        var=pd.DataFrame(index=adata.var_names),
        obs=pd.DataFrame(
            index=adata.obs_names, data={"__hvg_v3_batch_info__": batch_info}
        ),
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: We wrap the expression matrix in a small temporary AnnData whose only observation annotation is ``__hvg_v3_batch_info__``.
    # --- lecture ---
    # We wrap the expression matrix in a small temporary AnnData whose only
    # observation annotation is ``__hvg_v3_batch_info__``. That shell exists
    # so we can call scanpy's ``aggregate`` helper to compute per-batch mean
    # and variance with the same code path used elsewhere in the library.
    #
    # Nothing magical about the name: it is just an internal column we
    # control. The important idea is "same cells, same genes, labeled by
    # batch, ready for group-wise statistics."
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'Categorical', 'shape': (68579,), 'dtype': 'category'}
    #   norm_gene_vars = []
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None

    if batch_key is not None:
        # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
        # Alternate branch unused by test_lesson7b; no live 7B values here.
        # --- lecture ---
        # A real ``batch_key`` was provided, so we aggregate means and
        # variances within each batch. ``aggregate`` returns an AnnData-like
        # object with layers named ``mean`` and ``var``; for Dask inputs those
        # layers may still be lazy until we materialize them.
        #
        # Materialization here is intentional: the LOESS fit that follows is
        # a small client-side regression on per-gene summaries, not something
        # we want to leave tangled in a distributed graph.
        aggregated_mean_var = aggregate(
            adata_agg, by="__hvg_v3_batch_info__", func=["mean", "var"]
        )
        aggregated_mean_var.layers["mean"], aggregated_mean_var.layers["var"] = (
            materialize_as_ndarray(
                tuple(aggregated_mean_var.layers[l] for l in ["mean", "var"])
            )
        )
        # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
        # Alternate branch unused by test_lesson7b; no live 7B values here.
        # --- lecture ---
        # Batch-wise mean and variance layers are now concrete NumPy arrays in
        # memory. Each row of those layers corresponds to one batch and each
        # column to one gene. The upcoming loop will pick out the row for
        # batch ``b`` when it processes that batch's cells.
    else:
        # Single logical batch: reuse global means/variances.
        aggregated_mean_var = AnnData(
            var=pd.DataFrame(index=adata.var_names),
            obs=pd.DataFrame(
                index=np.array(["one"]), data={"__hvg_v3_batch_info__": np.array([0])}
            ),
            layers={
                "mean": df["means"].to_numpy().reshape((1, -1)),
                "var": df["variances"].to_numpy().reshape((1, -1)),
            },
        )
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: No batch key: we fabricate a one-row aggregated object that simply reuses the global mean and variance vectors already stored in ``df``.
        # --- lecture ---
        # No batch key: we fabricate a one-row aggregated object that simply
        # reuses the global mean and variance vectors already stored in
        # ``df``. The rest of the algorithm can use a single code path —
        # "for each batch in unique_batches" — even when there is only one
        # synthetic batch.
        #
        # That design avoids a special-case fork between "batched" and
        # "unbatched" for the LOESS and clipping stages.
        # --- facts at this step ---
        #   mean_layer_shape = (1, 20387)
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'Categorical', 'shape': (68579,), 'dtype': 'category'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'

    batch_info = batch_info.to_numpy()
    unique_batches = np.unique(batch_info)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: We are about to enter the per-batch loop.
    # --- lecture ---
    # We are about to enter the per-batch loop. For every distinct batch
    # label we will (1) select that batch's cells, (2) fit LOESS of
    # log-variance versus log-mean on the client, (3) build per-gene clip
    # thresholds, (4) call ``clip_square_sum`` on the batch's count matrix,
    # and (5) form a normalized variance vector for that batch.
    #
    # If the count matrix is Dask, step 4 may only build lazy graph pieces;
    # the heavy worker compute often waits until we ``da.compute`` all batch
    # results together after the loop.
    # --- facts at this step ---
    #   batches = [np.int64(0)]
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = []
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'

    for b in unique_batches:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: Starting one batch iteration.
        # --- lecture ---
        # Starting one batch iteration. Everything inside this loop is
        # "conditional on batch b": which cells we keep, which mean/variance
        # row we read, which clip thresholds we build, and which normalized
        # variance vector we append to ``norm_gene_vars``.
        #
        # If there is only one synthetic batch, this loop body still runs
        # exactly once and that is the entire analysis.
        # --- facts at this step ---
        #   batch = np.int64(0)
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'
        #   b = {'type': 'int64', 'shape': (), 'dtype': 'int64'}
        data_batch = data[batch_info == b]
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: ``data_batch`` is the expression matrix restricted to cells whose batch label equals ``b``.
        # --- lecture ---
        # ``data_batch`` is the expression matrix restricted to cells whose
        # batch label equals ``b``. For a NumPy array this is an ordinary
        # fancy-index or boolean take. For a Dask array it is usually another
        # lazy view: still chunked, still not necessarily loaded.
        #
        # Shape along axis 0 is the number of cells in this batch; axis 1 is
        # still all genes. Clip thresholds and LOESS fits are computed at
        # gene resolution, so keeping all columns is essential.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   batch_shape = (68579, 20387)
        #   batch_type = 'Array'
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'
        #   b = {'type': 'int64', 'shape': (), 'dtype': 'int64'}
        mean, var = (
            aggregated_mean_var[
                aggregated_mean_var.obs["__hvg_v3_batch_info__"] == b
            ].layers[l]
            for l in ["mean", "var"]
        )
        if isinstance(mean, CSBase):
            mean = mean.toarray()
            # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
            # Alternate branch unused by test_lesson7b; no live 7B values here.
            # --- lecture ---
            # The batch mean arrived as a sparse matrix (unusual but possible
            # depending on aggregation storage). LOESS and clipping want a
            # flat dense vector of length ``n_genes``, so we densify. With one
            # batch row this is tiny — not the full cells×genes matrix.
        mean = mean.ravel()
        if isinstance(var, CSBase):
            var = var.toarray()
            # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
            # Alternate branch unused by test_lesson7b; no live 7B values here.
            # --- lecture ---
            # Same densify-for-convenience step for the batch variance row.
            # After ``ravel`` we will have a 1-D NumPy vector suitable for
            # boolean masks like ``var > 0`` and for taking logarithms.
        var = var.ravel()
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: We now have aligned ``mean`` and ``var`` vectors for this batch — one entry per gene.
        # --- lecture ---
        # We now have aligned ``mean`` and ``var`` vectors for this batch —
        # one entry per gene. These are the inputs to the mean–variance trend
        # fit. Genes with zero variance are constant in the batch and cannot
        # sit on a log-variance plot, so they will be excluded from LOESS
        # momentarily.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   mean_len = 20387
        #   var_len = 20387
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'
        #   b = {'type': 'int64', 'shape': (), 'dtype': 'int64'}

        estimat_var = np.zeros(data.shape[1], dtype=np.float64)
        not_const = var > 0
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: ``not_const`` is a boolean mask of genes with positive variance in this batch.
        # --- lecture ---
        # ``not_const`` is a boolean mask of genes with positive variance in
        # this batch. Constant genes keep an estimated variance of zero in
        # ``estimat_var`` (our pre-allocated output). Non-constant genes will
        # receive LOESS fitted values on the log10(variance) scale.
        #
        # If somehow every gene were constant, we would skip the fit entirely;
        # in real single-cell data that almost never happens.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   n_not_const = 20387
        #   n_genes = 20387
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'
        #   b = {'type': 'int64', 'shape': (), 'dtype': 'int64'}
        if not_const.any():
            # Client-side loess fit of log10(var) ~ log10(mean).
            y = np.log10(var[not_const])
            x = np.log10(mean[not_const])
            # L7-LECTURE (real Lesson 7B run)
            # Step-by-step lecture note — inspected values from live execution.
            # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
            # topic: LOESS runs on the client, not on Dask workers.
            # --- lecture ---
            # LOESS runs on the client, not on Dask workers. We take log10 of
            # the batch means and variances for non-constant genes so the
            # trend is fit in log–log space, which is the usual scale for
            # mean–variance relationships in count data.
            #
            # ``span`` is the fraction of points used in each local fit: larger
            # span means a smoother, more global curve; smaller span follows
            # local wiggles more closely. Degree 2 means each local fit is
            # quadratic. After ``model.fit()``, ``fitted_values`` are the
            # estimated log10 variances along that smooth curve.
            # --- facts at this step ---
            #   batch = np.int64(0)
            #   span = 0.3
            #   n_points = 20387
            # --- locals / object fields at the call site ---
            #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
            #   flavor = 'seurat_v3'
            #   layer = 'counts'
            #   n_top_genes = 20387
            #   batch_key = None
            #   check_values = True
            #   span = 0.3
            #   subset = False
            #   inplace = False
            #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
            #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
            #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
            #   norm_gene_vars = []
            #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
            #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
            #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
            #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
            #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
            #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
            #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
            #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
            #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
            #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'
            #   b = {'type': 'int64', 'shape': (), 'dtype': 'int64'}
            model = loess(x, y, span=span, degree=2)
            model.fit()
            estimat_var[not_const] = model.outputs.fitted_values
            # L7-LECTURE (real Lesson 7B run)
            # Step-by-step lecture note — inspected values from live execution.
            # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
            # topic: The LOESS fitted log10 variances have been written back into ``estimat_var`` for every non-constant gene.
            # --- lecture ---
            # The LOESS fitted log10 variances have been written back into
            # ``estimat_var`` for every non-constant gene. Genes that were
            # constant remain zero. Next we undo the log10 to get a
            # regularized variance, then take a square root to get a
            # regularized standard deviation used for clipping and for the
            # normalized variance denominator.
            # --- facts at this step ---
            #   batch = np.int64(0)
            #   estimat_var_nonzero = 20387
            # --- locals / object fields at the call site ---
            #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
            #   flavor = 'seurat_v3'
            #   layer = 'counts'
            #   n_top_genes = 20387
            #   batch_key = None
            #   check_values = True
            #   span = 0.3
            #   subset = False
            #   inplace = False
            #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
            #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
            #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
            #   norm_gene_vars = []
            #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
            #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
            #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
            #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
            #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
            #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
            #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
            #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
            #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
            #   model = <_loess.loess object at 0x766de86723b0>
            #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'

        reg_std = np.sqrt(10**estimat_var)
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: ``reg_std`` is the regularized standard deviation per gene: sqrt(10 ** estimated_log10_variance).
        # --- lecture ---
        # ``reg_std`` is the regularized standard deviation per gene:
        # sqrt(10 ** estimated_log10_variance). It is "regularized" because
        # it comes from the smooth LOESS trend rather than from each gene's
        # raw sample standard deviation alone.
        #
        # Genes above the trend will later show large normalized variance;
        # genes on the trend look typical for their mean; genes below look
        # quieter than expected. The clip threshold also uses this
        # regularized scale so extreme counts are limited relative to the
        # expected noise level.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   reg_std_min = 0.00381771018001318
        #   reg_std_max = 15.01793167421097
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   aggregated_mean_var = AnnData object with n_obs × n_vars = 1 × 20387     obs: '__hvg_v3_batch_info__'     layers: 'mean', 'var'

        # Clip thresholds as in Seurat VST.
        n_obs = data_batch.shape[0]
        clip_val = reg_std * np.sqrt(n_obs) + mean
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: Seurat's VST clip threshold for each gene is ``mean + reg_std * sqrt(n_obs)``.
        # --- lecture ---
        # Seurat's VST clip threshold for each gene is
        # ``mean + reg_std * sqrt(n_obs)``. In words: start at the gene's
        # mean count level, then allow a generous number of regularized
        # standard deviations that grows with the square root of the number
        # of cells (as random-walk / CLT-style widths often do).
        #
        # Any individual cell's count above that ceiling will be clipped
        # before we accumulate sum and sum-of-squares. The thresholds are
        # computed here on the client as a simple NumPy vector; the actual
        # clipping of matrix entries happens inside ``clip_square_sum``,
        # which may run on workers for Dask inputs.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   n_obs = 68579
        #   clip_val_shape = (20387,)
        #   clip_val_head = [8.673160964700866, 2.0680641214392135, 0.9997809658065406]
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579

        # Lazy dask graph if data_batch is dask; compute happens later.
        squared_batch_counts_sum, batch_counts_sum = clip_square_sum(
            data_batch, clip_val
        )
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: ``clip_square_sum`` returned the two per-gene summaries for this batch.
        # --- lecture ---
        # ``clip_square_sum`` returned the two per-gene summaries for this
        # batch. If ``data_batch`` was a Dask array, these may still be lazy:
        # the graph knows how to clip and reduce each chunk, but workers may
        # not have run yet.
        #
        # If ``data_batch`` was already in memory, the summaries are concrete
        # NumPy arrays and the heavy lifting for this batch is done. Either
        # way, the next formula can treat them as "sum of squares" and "sum"
        # terms in the normalized variance algebra.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   squared_type = 'Array'
        #   sum_type = 'Array'
        #   squared_is_dask = True
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579

        norm_gene_var = (1 / ((n_obs - 1) * np.square(reg_std))) * (
            (n_obs * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: This expression rebuilds a clipped, variance-stabilized notion of per-gene variance and then divides by the squared regularized standard deviation (with the usual ``n_obs - 1`` Bessel factor in the denominator).
        # --- lecture ---
        # This expression rebuilds a clipped, variance-stabilized notion of
        # per-gene variance and then divides by the squared regularized
        # standard deviation (with the usual ``n_obs - 1`` Bessel factor in
        # the denominator). Expanding the algebra, the terms with
        # ``squared_batch_counts_sum`` and ``batch_counts_sum`` are exactly
        # the pieces you would get from summing ``(clipped_x - mean)²``
        # without forming the centered matrix explicitly.
        #
        # Keep the conceptual picture: we standardized relative to a smooth
        # mean–variance trend, after clipping outliers, and the resulting
        # ``norm_gene_var`` is what we rank to find HVGs.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   norm_type = 'Array'
        #   norm_is_dask = True
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = []
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579
        norm_gene_vars.append(norm_gene_var)
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: The normalized variance vector for this batch is appended to ``norm_gene_vars``.
        # --- lecture ---
        # The normalized variance vector for this batch is appended to
        # ``norm_gene_vars``. After the loop we will stack those vectors into
        # a batches×genes matrix, rank within each batch, and combine ranks
        # across batches according to the chosen flavor.
        # --- facts at this step ---
        #   batch = np.int64(0)
        #   n_collected = 1
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = [dask.array<multiply, shape=(20387,), dtype=float64, chunksize=(2000,), chunktype=numpy.ndarray>]
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579

    if any(isinstance(e, DaskArray) for e in norm_gene_vars):
        import dask.array as da

        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=1
        # topic: At least one batch's normalized variance is still a Dask array, which means the clip / sum-of-squares graph (and whatever else is fused into these expressions) has not fully executed.
        # --- lecture ---
        # At least one batch's normalized variance is still a Dask array, which
        # means the clip / sum-of-squares graph (and whatever else is fused
        # into these expressions) has not fully executed. Calling
        # ``da.compute`` on the whole list submits those tasks — gene-chunk
        # work to workers when using a distributed cluster — and blocks the
        # client until concrete NumPy results come back.
        #
        # This is the dramatic moment in Lesson 7: the scheduler fans out
        # ``per_block`` tasks to workers, and then the client continues with
        # in-memory arrays.
        # --- facts at this step ---
        #   n_arrays = 1
        #   types = ['Array']
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = [dask.array<multiply, shape=(20387,), dtype=float64, chunksize=(2000,), chunktype=numpy.ndarray>]
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579
        norm_gene_vars = da.compute(*norm_gene_vars)
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=1
        # topic: ``da.compute`` finished.
        # --- lecture ---
        # ``da.compute`` finished. Every entry of ``norm_gene_vars`` should
        # now be a real NumPy array with one value per gene. From here on the
        # algorithm is ordinary local NumPy/pandas: ranking, medians, sorting,
        # and writing annotations. No more distributed graph for this HVG
        # call.
        # --- facts at this step ---
        #   n_arrays = 1
        #   shapes = [(20387,)]
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = (array([0.890065  , 0.93527105, 1.00002917, ..., 0.88783959, 0.85511661,        0.91993357], shape=(20387,)),)
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579
    else:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=1
        # topic: None of the per-batch normalized variance vectors are Dask arrays, so there is nothing left to schedule.
        # --- lecture ---
        # None of the per-batch normalized variance vectors are Dask arrays,
        # so there is nothing left to schedule. We skip ``da.compute`` and
        # proceed directly to ranking. This is the typical path for eager
        # in-memory SciPy/NumPy inputs.
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
        #   data = {'type': 'csc_matrix', 'shape': (68579, 20387), 'dtype': 'float32', 'nnz': 37323295}
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = [array([0.890065  , 0.93527105, 1.00002917, ..., 0.88783959, 0.85511661,        0.91993357], shape=(20387,))]
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'csc_matrix', 'shape': (68579, 20387), 'dtype': 'float32', 'nnz': 37323295}
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766db8ac8430>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579
        pass

    norm_gene_vars = [ngv.reshape(1, -1) for ngv in norm_gene_vars]
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Each batch vector is reshaped to shape ``(1, n_genes)`` so we can concatenate along axis 0 into a 2-D array with one row per batch.
    # --- lecture ---
    # Each batch vector is reshaped to shape ``(1, n_genes)`` so we can
    # concatenate along axis 0 into a 2-D array with one row per batch.
    # Reshape does not change the numbers; it only sets up the stacking
    # layout.
    # --- facts at this step ---
    #   shapes = [(1, 20387)]
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = [array([[0.890065  , 0.93527105, 1.00002917, ..., 0.88783959, 0.85511661,         0.91993357]], shape=(1, 20387))]
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579
    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: ``norm_gene_vars`` is now a dense batches×genes matrix of normalized variances.
    # --- lecture ---
    # ``norm_gene_vars`` is now a dense batches×genes matrix of normalized
    # variances. Column ``j`` is gene ``j``; row ``i`` is batch ``i``. The
    # ranking stage will argsort within each row so that genes compete with
    # other genes inside the same batch before we combine evidence across
    # batches.
    # --- facts at this step ---
    #   shape = (1, 20387)
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579

    # argsort twice gives ranks; small rank = most variable.
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Double ``argsort`` is a standard trick to turn values into ranks.
    # --- lecture ---
    # Double ``argsort`` is a standard trick to turn values into ranks.
    # We negate first so that the largest normalized variance becomes rank 0
    # (most variable). After this, ``ranked_norm_gene_vars[b, g]`` is the
    # rank of gene ``g`` within batch ``b``.
    #
    # Analogy: line the genes up from "most spiky relative to the trend" to
    # "least spiky," and write their place in line. Small numbers are the
    # stars of that batch.
    # --- facts at this step ---
    #   ranked_shape = (1, 20387)
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579

    # SelectIntegrationFeatures-style bookkeeping.
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: For each gene we count how many batches placed it among the top ``n_top_genes`` ranks (ranks strictly less than ``n_top_genes``).
    # --- lecture ---
    # For each gene we count how many batches placed it among the top
    # ``n_top_genes`` ranks (ranks strictly less than ``n_top_genes``). That
    # count becomes ``highly_variable_nbatches``.
    #
    # A gene that is a top HVG in every batch gets a high count; a gene that
    # is only batch-specifically noisy gets a low count. Later, depending on
    # flavor, this count either breaks ties or is the primary sort key.
    # --- facts at this step ---
    #   n_top_genes = 20387
    #   nbatches_min = 1
    #   nbatches_max = 1
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Ranks at or beyond ``n_top_genes`` are set to NaN so they do not pull on the cross-batch median.
    # --- lecture ---
    # Ranks at or beyond ``n_top_genes`` are set to NaN so they do not pull
    # on the cross-batch median. The median rank across batches (ignoring
    # those NaNs) is the gene's consensus rank: low median rank means the
    # gene was repeatedly near the top of the HVG list.
    #
    # Genes that never made the top set in any batch can end up with NaN
    # median ranks and will sort to the end when we pick the final HVG set.
    # --- facts at this step ---
    #   median_ranked_finite = 20387
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 2), 'columns': ['means', 'variances']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579

    df = df.assign(
        gene_name=df.index,
        highly_variable_nbatches=num_batches_high_var,
        highly_variable_rank=median_ranked,
        variances_norm=np.mean(norm_gene_vars, axis=0),
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: The results table now has the Seurat-v3 annotation columns: ``highly_variable_nbatches``, ``highly_variable_rank`` (median rank), and ``variances_norm`` (mean of the per-batch normalized variances).
    # --- lecture ---
    # The results table now has the Seurat-v3 annotation columns:
    # ``highly_variable_nbatches``, ``highly_variable_rank`` (median rank),
    # and ``variances_norm`` (mean of the per-batch normalized variances).
    # We still have not marked the boolean ``highly_variable`` column; that
    # requires sorting by flavor-specific rules and taking the top N genes.
    # --- facts at this step ---
    #   columns = ['means', 'variances', 'gene_name', 'highly_variable_nbatches', 'highly_variable_rank', 'variances_norm']
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 6), 'columns': ['means', 'variances', 'gene_name', 'highly_variable_nbatches', 'highly_variable_rank', 'variances_norm']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579

    if flavor == "seurat_v3":
        sort_cols = ["highly_variable_rank", "highly_variable_nbatches"]
        sort_ascending = [True, False]
    elif flavor == "seurat_v3_paper":
        sort_cols = ["highly_variable_nbatches", "highly_variable_rank"]
        sort_ascending = [False, True]
    else:
        msg = f"Did not recognize flavor {flavor}"
        raise ValueError(msg)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Flavor controls the sort priority when choosing the final HVG list.
    # --- lecture ---
    # Flavor controls the sort priority when choosing the final HVG list.
    # ``seurat_v3`` prefers low median rank first, then breaks ties by
    # appearing in more batches. ``seurat_v3_paper`` prefers appearing in
    # more batches first, then breaks ties by low median rank — closer to
    # Seurat's SelectIntegrationFeatures bookkeeping.
    #
    # Same underlying statistics; different ways of asking "which genes are
    # reproducibly variable across batches?"
    # --- facts at this step ---
    #   flavor = 'seurat_v3'
    #   sort_cols = ['highly_variable_rank', 'highly_variable_nbatches']
    #   sort_ascending = [True, False]
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 6), 'columns': ['means', 'variances', 'gene_name', 'highly_variable_nbatches', 'highly_variable_rank', 'variances_norm']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579

    sorted_index = (
        df[sort_cols]
        .sort_values(sort_cols, ascending=sort_ascending, na_position="last")
        .index
    )
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: After sorting, the first ``n_top_genes`` gene names in ``sorted_index`` are labeled ``highly_variable=True``; everyone else stays False.
    # --- lecture ---
    # After sorting, the first ``n_top_genes`` gene names in ``sorted_index``
    # are labeled ``highly_variable=True``; everyone else stays False. That
    # boolean is what most downstream scanpy plots and subsetting operations
    # look at when people say "the HVGs."
    #
    # Note that ranks and normalized variances remain available for all genes,
    # not only the winners — the boolean is just the hard cutoff.
    # --- facts at this step ---
    #   n_top_genes = 20387
    #   n_hvg = 20387
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 7), 'columns': ['means', 'variances', 'gene_name', 'highly_variable_nbatches', 'highly_variable_rank', 'variances_norm...
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579

    if inplace:
        # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
        # Alternate branch unused by test_lesson7b; no live 7B values here.
        # --- lecture ---
        # ``inplace=True`` means we write the HVG annotations into
        # ``adata.var`` (and record the flavor under ``adata.uns['hvg']``)
        # instead of returning the DataFrame to the caller. This is the
        # common interactive scanpy style: after the call, ``adata`` itself
        # carries ``highly_variable``, ``means``, ``variances``, and friends.
        adata.uns["hvg"] = {"flavor": flavor}
        logg.hint(
            "added\n"
            "    'highly_variable', boolean vector (adata.var)\n"
            "    'highly_variable_rank', float vector (adata.var)\n"
            "    'means', float vector (adata.var)\n"
            "    'variances', float vector (adata.var)\n"
            "    'variances_norm', float vector (adata.var)"
        )
        for to_numpy_key in [
            "highly_variable",
            "highly_variable_rank",
            "means",
            "variances",
        ]:
            adata.var[to_numpy_key] = df[to_numpy_key].to_numpy()
        adata.var["variances_norm"] = (
            df["variances_norm"].to_numpy().astype("float64", copy=False)
        )
        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df[
                "highly_variable_nbatches"
            ].to_numpy()
        if subset:
            adata._inplace_subset_var(df["highly_variable"].to_numpy())
            # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
            # Alternate branch unused by test_lesson7b; no live 7B values here.
            # --- lecture ---
            # ``subset=True`` combined with ``inplace=True`` means we also
            # discard non-HVG genes from ``adata`` itself via an in-place
            # variable subset. After this, ``adata.n_vars`` equals the number
            # of selected highly variable genes. Use this when the rest of
            # the analysis should only see HVGs.
        # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
        # Alternate branch unused by test_lesson7b; no live 7B values here.
        # --- lecture ---
        # Inplace mode returns ``None`` because the results already live on
        # ``adata``. Callers that need a DataFrame should pass
        # ``inplace=False`` instead.
        return None

    if batch_key is None:
        df = df.drop(["highly_variable_nbatches"], axis=1)
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
        # topic: Without a batch key, the ``highly_variable_nbatches`` column is uninformative (every gene's count is 0 or 1 in a trivial way), so we drop it from the returned DataFrame to match the public API's expected columns for...
        # --- lecture ---
        # Without a batch key, the ``highly_variable_nbatches`` column is
        # uninformative (every gene's count is 0 or 1 in a trivial way), so
        # we drop it from the returned DataFrame to match the public API's
        # expected columns for the single-batch case.
        # --- locals / object fields at the call site ---
        #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
        #   flavor = 'seurat_v3'
        #   layer = 'counts'
        #   n_top_genes = 20387
        #   batch_key = None
        #   check_values = True
        #   span = 0.3
        #   subset = False
        #   inplace = False
        #   df = {'type': 'DataFrame', 'shape': (20387, 6), 'columns': ['means', 'variances', 'gene_name', 'highly_variable_rank', 'variances_norm', 'highly_variable']}
        #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
        #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
        #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
        #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
        #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
        #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
        #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
        #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   model = <_loess.loess object at 0x766de86723b0>
        #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
        #   n_obs = 68579
    if subset:
        df = df.iloc[df["highly_variable"].to_numpy(), :]
        # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
        # Alternate branch unused by test_lesson7b; no live 7B values here.
        # --- lecture ---
        # ``subset=True`` with ``inplace=False`` filters the returned
        # DataFrame down to rows where ``highly_variable`` is True. The
        # original ``adata`` is left unchanged; only the returned table is
        # shortened.

    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_highly_variable_genes_seurat_v3  where=scanpy  pid=2237068  hits_at_site=2
    # topic: Returning the results DataFrame to the caller.
    # --- lecture ---
    # Returning the results DataFrame to the caller. Each row is a gene; the
    # columns carry the Seurat-v3 metrics and the HVG boolean. This is the
    # non-inplace exit path — useful for tests and for pipelines that want
    # to manage annotations themselves.
    # --- facts at this step ---
    #   nrows = 20387
    #   columns = ['means', 'variances', 'gene_name', 'highly_variable_rank', 'variances_norm', 'highly_variable']
    # --- locals / object fields at the call site ---
    #   adata = AnnData object with n_obs × n_vars = 68579 × 20387     obs: 'batch'     var: 'gene_ids', 'n_cells'     layers: 'counts'
    #   flavor = 'seurat_v3'
    #   layer = 'counts'
    #   n_top_genes = 20387
    #   batch_key = None
    #   check_values = True
    #   span = 0.3
    #   subset = False
    #   inplace = False
    #   df = {'type': 'DataFrame', 'shape': (20387, 6), 'columns': ['means', 'variances', 'gene_name', 'highly_variable_rank', 'variances_norm', 'highly_variable']}
    #   data = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   batch_info = {'type': 'ndarray', 'shape': (68579,), 'dtype': 'int64'}
    #   norm_gene_vars = {'type': 'ndarray', 'shape': (1, 20387), 'dtype': 'float64'}
    #   adata_agg = AnnData object with n_obs × n_vars = 68579 × 20387     obs: '__hvg_v3_batch_info__'     layers: None
    #   unique_batches = {'type': 'ndarray', 'shape': (1,), 'dtype': 'int64'}
    #   data_batch = {'type': 'Array', 'shape': (68579, 20387), 'dtype': 'float32', 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'c...
    #   mean = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   var = {'type': 'ArrayView', 'shape': (20387,), 'dtype': 'float64'}
    #   estimat_var = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   not_const = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'bool'}
    #   y = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   x = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   model = <_loess.loess object at 0x766de86723b0>
    #   reg_std = {'type': 'ndarray', 'shape': (20387,), 'dtype': 'float64'}
    #   n_obs = 68579
    return df
