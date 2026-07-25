"""Shared helpers for the HVG CSC/dask tutorial suite.

Kept separate from ``conftest.py`` so both the fixtures and the individual test
modules can import the same loaders/comparators.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def narrate(where: str, title: str, **vars: Any) -> None:
    """Lesson-7 story log; enable with ``HVG_LESSON7_NARRATE=1``."""
    flag = os.environ.get("HVG_LESSON7_NARRATE", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    pid = os.getpid()
    detail = (" | " + " ".join(f"{k}={v!r}" for k, v in vars.items())) if vars else ""
    print(f"[L7 {where} pid={pid}] {title}{detail}", flush=True)


def _node(group: zarr.Group, key: str):
    """Navigate ``group`` by a slash-path like ``"X"`` or ``"layers/counts"``."""
    narrate("client", "_node: start path walk", key=key)
    node = group
    for part in key.split("/"):
        # Descend one zarr subgroup (e.g. layers → counts).
        node = node[part]
        narrate("client", "_node: descended", part=part, node_type=type(node).__name__)
    narrate("client", "_node: done", key=key)
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
    narrate("client", "load_lazy_csc: open store", store=str(store), key=key, gene_chunk=gene_chunk)
    group = zarr.open(str(store), mode="r")
    lazy = read_elem_lazy(_node(group, key))
    narrate(
        "client",
        "load_lazy_csc: read_elem_lazy returned",
        shape=lazy.shape,
        chunksize=lazy.chunksize,
        numblocks=lazy.numblocks,
    )
    meta = lazy._meta
    assert getattr(meta, "format", None) == "csc", (
        f"expected a CSC dask meta for {key!r}, got {type(meta).__name__}"
        f" (format={getattr(meta, 'format', None)!r})"
    )
    narrate("client", "load_lazy_csc: meta is CSC", meta_type=type(meta).__name__)
    chunked = lazy.rechunk((-1, gene_chunk))
    narrate(
        "client",
        "load_lazy_csc: after rechunk along genes",
        chunksize=chunked.chunksize,
        numblocks=chunked.numblocks,
    )
    assert chunked.chunksize[1] != chunked.shape[1], (
        "rechunk did not produce a column-chunked array: "
        f"chunksize={chunked.chunksize}, shape={chunked.shape}"
    )
    return chunked


def read_reference(store: Path | str) -> AnnData:
    """Fully in-memory AnnData reference (X + layers as scipy CSC)."""
    narrate("client", "read_reference: eager read_zarr", store=str(store))
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
    narrate(
        "client",
        "assert_hvg_close: compare result vs reference",
        result_shape=result.shape,
        reference_shape=reference.shape,
        atol=atol,
    )
    np.testing.assert_array_equal(result.index, reference.index)
    assert_frame_equal(result, reference, atol=atol, check_dtype=False)
    narrate("client", "assert_hvg_close: frames match within atol")


def list_gene_blocks(store: Path | str, key: str = "layers/counts") -> list[str]:
    """Sorted ``block_*`` names under ``key`` in a gene-block zarr store."""
    narrate("client", "list_gene_blocks: open store", store=str(store), key=key)
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    names = sorted(n for n in parent.keys() if str(n).startswith("block_"))
    narrate(
        "client",
        "list_gene_blocks: found block_* subgroups",
        n_blocks=len(names),
        names_head=names[:5],
    )
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
    narrate(
        "client",
        "assert_on_disk_gene_blocks: require multiple disk blocks",
        store=str(store),
        key=key,
        min_blocks=min_blocks,
    )
    names = list_gene_blocks(store, key)
    assert len(names) >= min_blocks, (
        f"expected >= {min_blocks} gene blocks under {key!r}, found {len(names)}: {names}"
    )
    narrate(
        "client",
        "assert_on_disk_gene_blocks: ok",
        n_blocks=len(names),
    )
    return names


def load_geneblock_csc(store: Path | str, key: str = "layers/counts"):
    """Load gene-block CSC subgroups as one column-chunked CSC dask array.

    Each ``block_XXX`` is read with ``read_elem_lazy`` and concatenated on the gene
    axis so ``numblocks[1]`` matches the number of on-disk subgroups (unlike
    ``load_lazy_csc``, which rechunks a single CSC blob only in memory).
    """
    narrate("client", "load_geneblock_csc: start", store=str(store), key=key)
    names = assert_on_disk_gene_blocks(store, key, min_blocks=2)
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    # read_elem_lazy may auto-split each CSC subgroup along genes (e.g. 1000);
    # rechunk so one dask column-chunk == one on-disk block_* subgroup.
    blocks = []
    for i, name in enumerate(names):
        narrate(
            "client",
            "load_geneblock_csc: lazy-read one on-disk gene block",
            i=i,
            name=name,
        )
        block = read_elem_lazy(parent[name])
        narrate(
            "client",
            "load_geneblock_csc: block after read_elem_lazy",
            name=name,
            shape=block.shape,
            chunksize=block.chunksize,
            numblocks=block.numblocks,
            meta_format=getattr(block._meta, "format", None),
        )
        assert getattr(block._meta, "format", None) == "csc", (
            f"{key}/{name}: expected CSC dask meta, got {type(block._meta).__name__}"
        )
        # Force a single column chunk spanning this block's genes.
        rechunked = block.rechunk((-1, block.shape[1]))
        narrate(
            "client",
            "load_geneblock_csc: rechunked to one gene chunk per disk block",
            name=name,
            shape=rechunked.shape,
            chunksize=rechunked.chunksize,
            numblocks=rechunked.numblocks,
        )
        blocks.append(rechunked)

    narrate(
        "client",
        "load_geneblock_csc: concatenate along gene axis",
        n_blocks=len(blocks),
    )
    combined = da.concatenate(blocks, axis=1)
    narrate(
        "client",
        "load_geneblock_csc: concatenated dask CSC",
        shape=combined.shape,
        numblocks=combined.numblocks,
        chunksize=combined.chunksize,
    )
    assert combined.numblocks[1] == len(names), (
        f"expected {len(names)} column blocks, got {combined.numblocks}"
    )
    assert combined.chunksize[1] != combined.shape[1], (
        "concatenated array is not column-chunked: "
        f"chunksize={combined.chunksize}, shape={combined.shape}"
    )
    narrate(
        "client",
        "load_geneblock_csc: done — column chunks match on-disk blocks",
        n_disk_blocks=len(names),
        numblocks=combined.numblocks,
    )
    return combined


def load_geneblock_csc_memory(store: Path | str, key: str = "layers/counts"):
    """Eager scipy CSC reference: hstack of the same on-disk gene blocks."""
    narrate("compare", "load_geneblock_csc_memory: eager path start", store=str(store))
    names = list_gene_blocks(store, key)
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    pieces = []
    for name in names:
        # Eager read_elem → scipy matrix (no dask).
        mat = csc_matrix(read_elem(parent[name]))
        narrate(
            "compare",
            "load_geneblock_csc_memory: loaded one block into memory",
            name=name,
            shape=mat.shape,
            nnz=mat.nnz,
        )
        pieces.append(mat)
    out = hstack(pieces, format="csc")
    narrate(
        "compare",
        "load_geneblock_csc_memory: hstack complete",
        shape=out.shape,
        nnz=out.nnz,
    )
    return out
