"""Shared helpers for the HVG CSC/dask tutorial suite.

Kept separate from ``conftest.py`` so both the fixtures and the individual test
modules can import the same loaders/comparators.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Resolve local forks whether this suite lives at scverse/learn/... or scanpy/learn/...
_HERE = Path(__file__).resolve().parent
_SCVERSE = _HERE.parents[1]  # scverse/learn/hvg_csc_dask
_SCANPY = _HERE.parents[2]  # scanpy/learn/hvg_csc_dask
if (_SCVERSE / "scanpy" / "src").is_dir():
    _SRCS = (_SCVERSE / "anndata" / "src", _SCVERSE / "scanpy" / "src")
else:
    _SRCS = (_SCANPY.parent / "anndata" / "src", _SCANPY / "src")
for _src in _SRCS:
    _src_str = str(_src)
    if _src.is_dir() and _src_str not in sys.path:
        sys.path.insert(0, _src_str)

import dask.array as da  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from anndata import read_zarr  # noqa: E402
from anndata.experimental import read_elem_lazy  # noqa: E402
from anndata.io import read_elem  # noqa: E402
from pandas.testing import assert_frame_equal  # noqa: E402
from scipy.sparse import csc_matrix, hstack  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData

DEFAULT_GENE_CHUNK = 200


def _node(group: zarr.Group, key: str):
    """Navigate ``group`` by a slash-path like ``"X"`` or ``"layers/counts"``."""
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_node  where=client  pid=2237068  hits_at_site=5
    # topic: We are about to walk into a Zarr store the way you walk into a nested set of folders on disk.
    # --- lecture ---
    # We are about to walk into a Zarr store the way you walk into a nested
    # set of folders on disk. AnnData does not store one giant flat file for
    # everything; it stores a hierarchy. A path like "layers/counts" means:
    # open the group named "layers", then inside it open the group named
    # "counts".
    #
    # Think of Zarr as a filing cabinet. The root is the cabinet. Each name
    # between slashes is a drawer or a folder inside a drawer. We are not
    # loading the gene expression numbers yet — we are only finding the
    # correct drawer so later code can read what is inside it.
    #
    # Why do this with a helper? Because the same slash-path idea shows up
    # everywhere in AnnData I/O. Once you can picture "layers/counts" as
    # two dictionary-like lookups in a row, the rest of the lesson becomes
    # much less mysterious.
    # --- facts at this step ---
    #   key = 'layers/counts'
    # --- locals / object fields at the call site ---
    #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
    #   key = 'layers/counts'
    node = group
    for part in key.split("/"):
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_node  where=client  pid=2237068  hits_at_site=10
        # topic: Next we take one step deeper into the Zarr hierarchy: we look up the child named ….
        # --- lecture ---
        # Next we take one step deeper into the Zarr hierarchy: we look up
        # the child named {…}.
        #
        # In Python terms this is almost exactly like writing
        # parent[{…}]. In Zarr terms, that child might be another group
        # (a folder) or an array (the actual numbers). For gene-block stores,
        # "layers" and "counts" are groups; the individual "block_000"
        # children under counts are themselves small sparse-matrix groups.
        #
        # After this step, "node" no longer means the whole store — it means
        # "wherever we are after opening {…}". The next iteration of the
        # loop (if any) continues from here.
        # --- facts at this step ---
        #   part = 'layers'
        #   node_type_before = 'Group'
        # --- locals / object fields at the call site ---
        #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   key = 'layers/counts'
        #   node = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   part = 'layers'
        node = node[part]
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=_node  where=client  pid=2237068  hits_at_site=10
        # topic: We successfully opened …. The object we hold now is a Zarr node of type ….
        # --- lecture ---
        # We successfully opened {…}. The object we hold now is a Zarr
        # node of type {…}.
        #
        # If that type is a Group, we can still go deeper or list its
        # children. If it were an Array, we would already be at leaf data.
        # For Lesson 7 we usually stop at the parent of the block_* groups
        # so we can list those blocks next.
        # --- facts at this step ---
        #   part = 'layers'
        #   node_type = 'Group'
        # --- locals / object fields at the call site ---
        #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   key = 'layers/counts'
        #   node = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers>
        #   part = 'layers'
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=_node  where=client  pid=2237068  hits_at_site=5
    # topic: The path walk is finished.
    # --- lecture ---
    # The path walk is finished. We now have a handle on the Zarr node that
    # the slash-path pointed to. Nothing about the count matrix has been
    # computed yet — we only located the place on disk (or in the store)
    # where those counts live.
    #
    # Next steps in the callers will either list block_* children under this
    # node, or ask AnnData to read one of those children lazily as a dask
    # array.
    # --- facts at this step ---
    #   key = 'layers/counts'
    #   final_type = 'Group'
    # --- locals / object fields at the call site ---
    #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
    #   key = 'layers/counts'
    #   node = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
    #   part = 'counts'
    return node


def load_lazy_csc(
    store: Path | str, key: str, *, gene_chunk: int = DEFAULT_GENE_CHUNK
):
    """Read ``key`` from the zarr store as a **column-chunked CSC dask array**.

    "Column-chunked" is produced here, not on disk: the store holds a single CSC
    component, ``read_elem_lazy`` hands back a dask array with a CSC ``_meta``,
    and ``.rechunk((-1, gene_chunk))`` splits it along the gene axis so it hits
    the code path our HVG changes target.
    """
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # This helper is the "older" Lesson 1–6 style loader. The store holds ONE
    # big CSC sparse matrix for the key (for example X or layers/counts). We
    # will open that matrix lazily, then artificially split it along genes
    # with .rechunk so scanpy sees multiple column chunks.
    #
    # Contrast this with Lesson 7's load_geneblock_csc: there, each gene
    # chunk is already a separate on-disk subgroup. Here, the split is only
    # in the dask graph in memory — the files on disk are still one CSC
    # blob.
    #
    # Opening the store in mode "r" means read-only. We are not modifying
    # anything; we are preparing a recipe for later computation.
    group = zarr.open(str(store), mode="r")
    lazy = read_elem_lazy(_node(group, key))
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # AnnData's read_elem_lazy has returned a Dask array whose "meta"
    # (a tiny example object) is a scipy CSC matrix. That meta is how Dask
    # knows "when I eventually compute a chunk, the result should behave
    # like CSC sparse data", even though right now almost no numbers have
    # been read into RAM.
    #
    # The shape and chunksize you see in the facts below describe the lazy
    # layout AnnData chose by default (often chunking the major axis of
    # CSC, which is genes/columns). We are about to rechunk so that each
    # chunk has a controlled number of genes — that is what gene_chunk
    # means.
    meta = lazy._meta
    assert getattr(meta, "format", None) == "csc", (
        f"expected a CSC dask meta for {key!r}, got {type(meta).__name__}"
        f" (format={getattr(meta, 'format', None)!r})"
    )
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # We just asserted that the lazy array really is CSC-in-dask. CSC means
    # Compressed Sparse Column: the matrix stores non-zero entries organized
    # by column (gene). That matches how highly variable gene methods like
    # to walk down genes.
    #
    # If this were CSR (row-compressed), column-wise gene chunks would be the
    # awkward direction. The whole Lesson 7 story is built on CSC being the
    # natural format for "give me all cells for a subset of genes".
    chunked = lazy.rechunk((-1, gene_chunk))
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # .rechunk((-1, gene_chunk)) means: keep the entire observation (cell)
    # axis in one piece (-1), and split the gene axis into pieces of about
    # gene_chunk columns each.
    #
    # After this, Dask can schedule work per gene-block. Important subtlety:
    # for a single on-disk CSC blob, rechunking does not magically create
    # separate files. It creates a logical graph. Lesson 7 goes further by
    # writing those gene blocks as real subgroups on disk.
    assert chunked.chunksize[1] != chunked.shape[1], (
        "rechunk did not produce a column-chunked array: "
        f"chunksize={chunked.chunksize}, shape={chunked.shape}"
    )
    return chunked


def read_reference(store: Path | str) -> AnnData:
    """Fully in-memory AnnData reference (X + layers as scipy CSC)."""
    # L7-LECTURE (not hit on Lesson 7B primary gene-block path)
    # Alternate branch unused by test_lesson7b; no live 7B values here.
    # --- lecture ---
    # read_reference loads the entire AnnData from Zarr into memory at once.
    # That is the opposite of the lazy Lesson 7 path: here we pay the RAM
    # cost up front and get ordinary scipy/numpy-backed arrays.
    #
    # We use eager loads when we need a simple ground truth, not when we are
    # practising out-of-core parallelism.
    return read_zarr(str(store))


def n_column_chunks(store: Path | str, key: str, *, gene_chunk: int) -> int:
    return len(load_lazy_csc(store, key, gene_chunk=gene_chunk).chunks[1])


def assert_hvg_close(
    result: pd.DataFrame, reference: pd.DataFrame, *, atol: float = 1e-4
) -> None:
    """Compare two ``highly_variable_genes`` result frames tolerantly.

    Float columns are compared with ``atol``; ``check_dtype=False`` because dask
    reductions can promote dtypes. Callers should select every gene
    (``n_top_genes=n_vars``) so the discrete ``highly_variable`` boolean does not
    flip on sub-``atol`` differences at the top-N boundary.
    """
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=assert_hvg_close  where=client  pid=2237068  hits_at_site=1
    # topic: We are comparing two highly_variable_genes result tables: one from the lazy/dask path and one from an in-memory reference.
    # --- lecture ---
    # We are comparing two highly_variable_genes result tables: one from the
    # lazy/dask path and one from an in-memory reference. Both should
    # describe the same genes (same index) and nearly the same numeric
    # metrics.
    #
    # Floating-point reductions over parallel chunks can differ by tiny
    # amounts from a single-threaded sum, so we allow a small absolute
    # tolerance (atol). We also ignore exact dtype matches, because dask
    # sometimes promotes floats.
    #
    # If this assertion fails, either the lazy path computed a different
    # scientific answer, or the tolerance is too tight for this dataset.
    # --- facts at this step ---
    #   result_shape = (20387, 6)
    #   reference_shape = (20387, 6)
    #   atol = 0.0001
    # --- locals / object fields at the call site ---
    #   result = {'type': 'DataFrame', 'shape': (20387, 6)}
    #   reference = {'type': 'DataFrame', 'shape': (20387, 6)}
    #   atol = 0.0001
    np.testing.assert_array_equal(result.index, reference.index)
    assert_frame_equal(result, reference, atol=atol, check_dtype=False)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=assert_hvg_close  where=client  pid=2237068  hits_at_site=1
    # topic: Good news: the dask/out-of-core answer and the eager in-memory answer agree within tolerance.
    # --- lecture ---
    # Good news: the dask/out-of-core answer and the eager in-memory answer
    # agree within tolerance. That is the scientific sanity check for this
    # lesson — parallelism and lazy reading did not change the biology of
    # the result, only how the work was scheduled and how data was paged
    # from disk.
    # --- locals / object fields at the call site ---
    #   result = {'type': 'DataFrame', 'shape': (20387, 6)}
    #   reference = {'type': 'DataFrame', 'shape': (20387, 6)}
    #   atol = 0.0001


def list_gene_blocks(store: Path | str, key: str = "layers/counts") -> list[str]:
    """Sorted ``block_*`` names under ``key`` in a gene-block zarr store."""
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=list_gene_blocks  where=client  pid=2237068  hits_at_site=3
    # topic: Lesson 7 stores do not keep counts as one CSC object.
    # --- lecture ---
    # Lesson 7 stores do not keep counts as one CSC object. Under
    # layers/counts you will find siblings named block_000, block_001, and
    # so on. Each block is a CSC matrix covering ALL cells but only a
    # contiguous slice of genes.
    #
    # Imagine a spreadsheet with thousands of gene columns. Instead of one
    # enormous file, we tore the spreadsheet into vertical strips and filed
    # each strip separately. list_gene_blocks is simply reading the labels
    # on those strips from the filing cabinet.
    #
    # Sorting the names keeps block order aligned with gene order from left
    # to right, which matters when we concatenate them back into one logical
    # matrix.
    # --- facts at this step ---
    #   store = '/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr'
    #   key = 'layers/counts'
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    names = sorted(n for n in parent.keys() if str(n).startswith("block_"))
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=list_gene_blocks  where=client  pid=2237068  hits_at_site=3
    # topic: We listed every child under the counts group whose name starts with "block_".
    # --- lecture ---
    # We listed every child under the counts group whose name starts with
    # "block_". Those names are our on-disk gene chunks.
    #
    # If this list were empty, the store would not be a Lesson 7 gene-block
    # store at all — it might be a normal AnnData Zarr from Lessons 0–6.
    # The test requires multiple blocks so we can demonstrate real
    # column-wise parallelism.
    # --- facts at this step ---
    #   n_blocks = 11
    #   names_head = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007']
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
    #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    if not names:
        msg = f"no block_* subgroups under {key!r} in {store}"
        raise FileNotFoundError(msg)
    return names


def assert_on_disk_gene_blocks(
    store: Path,
    key: str = "layers/counts",
    min_blocks: int = 2,
) -> list[str]:
    """Assert the store has multiple on-disk CSC gene-block subgroups."""
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=assert_on_disk_gene_blocks  where=client  pid=2237068  hits_at_site=1
    # topic: Before we invest time in LocalCluster and HVG, we verify the store really has the Lesson 7 layout: at least min_blocks separate gene strips on disk.
    # --- lecture ---
    # Before we invest time in LocalCluster and HVG, we verify the store
    # really has the Lesson 7 layout: at least min_blocks separate gene
    # strips on disk.
    #
    # Why insist on more than one block? With a single block there is nothing
    # to parallelize along genes — you would be back to "one big matrix"
    # even if the folder names look fancy. Multiple blocks are what make
    # "workers pull different gene ranges" a meaningful story.
    # --- facts at this step ---
    #   store = '/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr'
    #   key = 'layers/counts'
    #   min_blocks = 2
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    #   min_blocks = 2
    names = list_gene_blocks(store, key)
    assert len(names) >= min_blocks, (
        f"expected >= {min_blocks} gene blocks under {key!r}, found {len(names)}: {names}"
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=assert_on_disk_gene_blocks  where=client  pid=2237068  hits_at_site=1
    # topic: The on-disk layout check passed.
    # --- lecture ---
    # The on-disk layout check passed. We now know how many gene-block
    # subgroups exist, and we will soon insist that the lazy dask array we
    # build has exactly that many column chunks — one dask chunk per disk
    # strip.
    # --- facts at this step ---
    #   n_blocks = 11
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    #   min_blocks = 2
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    return names


def load_geneblock_csc(store: Path | str, key: str = "layers/counts"):
    """Load gene-block CSC subgroups as one column-chunked CSC dask array.

    Each ``block_XXX`` is read with ``read_elem_lazy`` and concatenated on the
    gene axis so ``numblocks[1]`` matches the number of on-disk subgroups.
    """
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=load_geneblock_csc  where=client  pid=2237068  hits_at_site=1
    # topic: This is the heart of the Lesson 7 data path.
    # --- lecture ---
    # This is the heart of the Lesson 7 data path.
    #
    # Goal: build ONE logical cells×genes matrix that scanpy can treat as a
    # dask array, while each gene chunk still corresponds to a real on-disk
    # CSC subgroup (block_000, block_001, …).
    #
    # Plan: for each block name, call AnnData's read_elem_lazy (which returns
    # a dask array without reading all values yet), force that piece to be a
    # single column chunk, then concatenate the pieces side-by-side along
    # the gene axis.
    #
    # Analogy: each block is a vertical strip of the spreadsheet stored in
    # its own envelope. We are lining the envelopes up in order and telling
    # Dask "this is one wide spreadsheet made of these strips", without
    # opening every envelope into RAM right now.
    #
    # (If the test already listed block_* names a moment ago, you will see
    # those names again here — this function always re-reads the directory
    # so it can stand alone when called from other lessons.)
    # --- facts at this step ---
    #   store = '/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr'
    #   key = 'layers/counts'
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    names = list_gene_blocks(store, key)
    if len(names) < 2:
        msg = f"expected >= 2 gene blocks under {key!r}, found {len(names)}: {names}"
        raise AssertionError(msg)
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    blocks = []
    for i, name in enumerate(names):
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=load_geneblock_csc  where=client  pid=2237068  hits_at_site=11
        # topic: We are loading on-disk gene strip 1 named 'block_001'.
        # --- lecture ---
        # We are loading on-disk gene strip {…} named {…}.
        #
        # read_elem_lazy will look at the encoding metadata on that Zarr
        # group (encoding-type: csc_matrix, shape, etc.) and construct a
        # Dask array whose tasks know how to read slices of that sparse
        # matrix later. Importantly, the heavy numeric data is still on
        # disk after this call returns.
        #
        # AnnData may initially choose its own internal chunk sizes for a
        # single CSC object (for example splitting along genes every 1000
        # columns). That is fine for one file, but for Lesson 7 we want a
        # simpler contract: after we rechunk, THIS disk block becomes
        # exactly ONE dask column chunk. That way numblocks along genes
        # equals the number of block_* folders.
        # --- facts at this step ---
        #   i = 1
        #   name = 'block_001'
        #   n_total = 11
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   key = 'layers/counts'
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
        #   blocks = [dask.array<rechunk-merge, shape=(68579, 2000), dtype=float32, chunksize=(68579, 2000), chunktype=scipy.csc_matrix>]
        #   i = 1
        #   name = 'block_001'
        #   block = {'type': 'Array', 'shape': (68579, 2000), 'numblocks': (1, 2), 'chunksize': (68579, 1000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        #   rechunked = {'type': 'Array', 'shape': (68579, 2000), 'numblocks': (1, 1), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        block = read_elem_lazy(parent[name])
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=load_geneblock_csc  where=client  pid=2237068  hits_at_site=11
        # topic: Lazy read of … succeeded. Look at the facts: you should see a shape like (n_cells, n_genes_in_this_strip), a CSC meta format, and some chunksize that AnnData chose.
        # --- lecture ---
        # Lazy read of {…} succeeded. Look at the facts: you should see
        # a shape like (n_cells, n_genes_in_this_strip), a CSC meta format,
        # and some chunksize that AnnData chose.
        #
        # Next we call .rechunk((-1, block.shape[1])). In words: "one chunk
        # tall enough to cover all cells, and wide enough to cover every
        # gene that lives in this strip". After that, this strip is atomic
        # from Dask's point of view — a natural unit of work for a worker.
        # --- facts at this step ---
        #   name = 'block_001'
        #   shape = (68579, 2000)
        #   chunksize = (68579, 1000)
        #   numblocks = (1, 2)
        #   meta_format = 'csc'
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   key = 'layers/counts'
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
        #   blocks = [dask.array<rechunk-merge, shape=(68579, 2000), dtype=float32, chunksize=(68579, 2000), chunktype=scipy.csc_matrix>]
        #   i = 1
        #   name = 'block_001'
        #   block = {'type': 'Array', 'shape': (68579, 2000), 'numblocks': (1, 2), 'chunksize': (68579, 1000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        #   rechunked = {'type': 'Array', 'shape': (68579, 2000), 'numblocks': (1, 1), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        assert getattr(block._meta, "format", None) == "csc", (
            f"{key}/{name}: expected CSC dask meta, got {type(block._meta).__name__}"
        )
        rechunked = block.rechunk((-1, block.shape[1]))
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=load_geneblock_csc  where=client  pid=2237068  hits_at_site=11
        # topic: Rechunk of … is done. This piece should now report numblocks == (1, 1): one block of cells × one block of genes for this strip.
        # --- lecture ---
        # Rechunk of {…} is done. This piece should now report
        # numblocks == (1, 1): one block of cells × one block of genes for
        # this strip.
        #
        # We append it to a Python list. After the loop we will concatenate
        # the list along axis=1 (genes), reconstructing the full width of
        # the original count matrix as a single dask array.
        # --- facts at this step ---
        #   name = 'block_000'
        #   shape = (68579, 2000)
        #   chunksize = (68579, 2000)
        #   numblocks = (1, 1)
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   key = 'layers/counts'
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
        #   blocks = []
        #   i = 0
        #   name = 'block_000'
        #   block = {'type': 'Array', 'shape': (68579, 2000), 'numblocks': (1, 2), 'chunksize': (68579, 1000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        #   rechunked = {'type': 'Array', 'shape': (68579, 2000), 'numblocks': (1, 1), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
        blocks.append(rechunked)

    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=load_geneblock_csc  where=client  pid=2237068  hits_at_site=1
    # topic: All strips are in the list as lazy dask arrays.
    # --- lecture ---
    # All strips are in the list as lazy dask arrays. da.concatenate(...,
    # axis=1) glues them left-to-right along genes.
    #
    # The result is still lazy. Concatenate builds a larger graph: "to get
    # column chunk k, read and compute strip k". No worker has been asked
    # to touch the bytes yet. That happens later when scanpy's seurat_v3
    # code calls da.compute (or when mean/var reductions compute).
    # --- facts at this step ---
    #   n_blocks = 11
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
    #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
    #   blocks = [dask.array<rechunk-merge, shape=(68579, 2000), dtype=float32, chunksize=(68579, 2000), chunktype=scipy.csc_matrix>, dask.array<rechunk-merge, shape=(68579, ...
    #   i = 10
    #   name = 'block_010'
    #   block = {'type': 'Array', 'shape': (68579, 387), 'numblocks': (1, 1), 'chunksize': (68579, 387), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    #   rechunked = {'type': 'Array', 'shape': (68579, 387), 'numblocks': (1, 1), 'chunksize': (68579, 387), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    combined = da.concatenate(blocks, axis=1)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=load_geneblock_csc  where=client  pid=2237068  hits_at_site=1
    # topic: Concatenation produced the full logical matrix.
    # --- lecture ---
    # Concatenation produced the full logical matrix. Check the facts
    # carefully:
    #
    # • shape should be (all cells, all genes)
    # • numblocks[1] should equal the number of on-disk block_* groups
    # • chunksize[1] should be smaller than the total number of genes
    #   (otherwise you only have one column chunk and the lesson failed)
    #
    # This object is what we will place into adata.layers['counts'] before
    # calling highly_variable_genes.
    #
    # Lecture analysis of the live Lesson 7B state:
    #   • combined.shape = (68579, 20387): full Fresh-68k filtered matrix.
    #   • combined.numblocks = (1, 11): one obs chunk, eleven gene chunks —
    #     matching names = block_000 … block_010.
    #   • combined.chunksize = (68579, 2000): typical strip width 2000 genes;
    #     last strip block_010 was narrower: (68579, 387).
    #   • store = .../pbmc68k_geneblocks.zarr (not the smaller pbmc3k store).
    #   • meta_format='csc': still CSC-in-dask after concatenate.
    #
    # --- facts at this step ---
    #   shape = (68579, 20387)
    #   numblocks = (1, 11)
    #   chunksize = (68579, 2000)
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    #   names = ['block_000', ..., 'block_010']  # 11 strips
    #   parent = <Group .../pbmc68k_geneblocks.zarr/layers/counts>
    #   i = 10 / name = 'block_010'
    #   block/rechunked last strip = (68579, 387) CSC dask
    #   combined = (68579, 20387) numblocks=(1, 11) chunksize=(68579, 2000) CSC
    assert combined.numblocks[1] == len(names), (
        f"expected {len(names)} column blocks, got {combined.numblocks}"
    )
    assert combined.chunksize[1] != combined.shape[1], (
        "concatenated array is not column-chunked: "
        f"chunksize={combined.chunksize}, shape={combined.shape}"
    )
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=load_geneblock_csc  where=client  pid=2237068  hits_at_site=1
    # topic: Layout assertions passed.
    # --- lecture ---
    # Layout assertions passed. We have a column-chunked CSC dask array
    # whose chunks line up with real gene-block files. That is the
    # precondition for out-of-core, gene-parallel seurat_v3 under a
    # LocalCluster.
    # --- facts at this step ---
    #   n_disk_blocks = 11
    #   numblocks = (1, 11)
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
    #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
    #   blocks = [dask.array<rechunk-merge, shape=(68579, 2000), dtype=float32, chunksize=(68579, 2000), chunktype=scipy.csc_matrix>, dask.array<rechunk-merge, shape=(68579, ...
    #   i = 10
    #   name = 'block_010'
    #   block = {'type': 'Array', 'shape': (68579, 387), 'numblocks': (1, 1), 'chunksize': (68579, 387), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    #   rechunked = {'type': 'Array', 'shape': (68579, 387), 'numblocks': (1, 1), 'chunksize': (68579, 387), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    #   combined = {'type': 'Array', 'shape': (68579, 20387), 'numblocks': (1, 11), 'chunksize': (68579, 2000), 'meta_type': 'csc_matrix', 'meta_format': 'csc'}
    return combined


def load_geneblock_csc_memory(store: Path | str, key: str = "layers/counts"):
    """Eager scipy CSC reference: hstack of the same on-disk gene blocks."""
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=load_geneblock_csc_memory  where=compare  pid=2237068  hits_at_site=1
    # topic: This is the control experiment.
    # --- lecture ---
    # This is the control experiment. We read the SAME on-disk gene blocks,
    # but eagerly: each block becomes a real scipy CSC matrix in RAM, then
    # we hstack them into one big in-memory CSC matrix.
    #
    # There is no Dask graph here and no worker pool. If seurat_v3 on this
    # matrix agrees with seurat_v3 on the lazy concatenated matrix, we gain
    # confidence that the parallel path is scientifically faithful.
    #
    # Cost: for large datasets this control can use a lot of memory. That is
    # why it is the "compare" path, not the main Lesson 7 demo path.
    # --- facts at this step ---
    #   store = '/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr'
    #   key = 'layers/counts'
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    names = list_gene_blocks(store, key)
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    pieces = []
    for name in names:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=load_geneblock_csc_memory  where=compare  pid=2237068  hits_at_site=11
        # topic: Eagerly reading 'block_001' with read_elem (not read_elem_lazy).
        # --- lecture ---
        # Eagerly reading {…} with read_elem (not read_elem_lazy).
        # That pulls the sparse arrays (data, indices, indptr) into memory
        # and builds a scipy matrix immediately.
        #
        # We wrap/ensure CSC, then keep the piece for hstack. Repeating this
        # for every block reconstructs the full matrix without dask.
        # --- facts at this step ---
        #   name = 'block_001'
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   key = 'layers/counts'
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
        #   pieces = [<Compressed Sparse Column sparse matrix of dtype 'float32' 	with 3551519 stored elements and shape (68579, 2000)>]
        #   name = 'block_001'
        #   mat = {'type': 'csc_matrix', 'shape': (68579, 2000), 'nnz': 3551519}
        mat = csc_matrix(read_elem(parent[name]))
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=load_geneblock_csc_memory  where=compare  pid=2237068  hits_at_site=11
        # topic: Block 'block_000' is now a concrete scipy CSC matrix in this process.
        # --- lecture ---
        # Block {…} is now a concrete scipy CSC matrix in this process.
        # nnz (number of stored non-zeros) tells you how sparse this strip
        # is. Dense would be n_cells * n_genes_in_block entries; sparse
        # stores only the non-zeros.
        # --- facts at this step ---
        #   name = 'block_000'
        #   shape = (68579, 2000)
        #   nnz = 3551519
        # --- locals / object fields at the call site ---
        #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
        #   key = 'layers/counts'
        #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
        #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
        #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
        #   pieces = []
        #   name = 'block_000'
        #   mat = {'type': 'csc_matrix', 'shape': (68579, 2000), 'nnz': 3551519}
        pieces.append(mat)
    out = hstack(pieces, format="csc")
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=load_geneblock_csc_memory  where=compare  pid=2237068  hits_at_site=1
    # topic: hstack stacked every gene strip into one wide CSC matrix.
    # --- lecture ---
    # hstack stacked every gene strip into one wide CSC matrix. This is the
    # reference object for full_compare in the Lesson 7 tests.
    # --- facts at this step ---
    #   shape = (68579, 20387)
    #   nnz = 37323295
    # --- locals / object fields at the call site ---
    #   store = PosixPath('/home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr')
    #   key = 'layers/counts'
    #   names = ['block_000', 'block_001', 'block_002', 'block_003', 'block_004', 'block_005', 'block_006', 'block_007', 'block_008', 'block_009', 'block_010']
    #   group = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr>
    #   parent = <Group file:///home/jonathan/scverse/learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr/layers/counts>
    #   pieces = [<Compressed Sparse Column sparse matrix of dtype 'float32' 	with 3551519 stored elements and shape (68579, 2000)>, <Compressed Sparse Column sparse matrix o...
    #   name = 'block_010'
    #   mat = {'type': 'csc_matrix', 'shape': (68579, 387), 'nnz': 1041421}
    #   out = {'type': 'csc_matrix', 'shape': (68579, 20387), 'nnz': 37323295}
    return out
