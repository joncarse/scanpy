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
    node = group
    for part in key.split("/"):
        node = node[part]
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
    group = zarr.open(str(store), mode="r")
    lazy = read_elem_lazy(_node(group, key))
    meta = lazy._meta
    assert getattr(meta, "format", None) == "csc", (
        f"expected a CSC dask meta for {key!r}, got {type(meta).__name__}"
        f" (format={getattr(meta, 'format', None)!r})"
    )
    chunked = lazy.rechunk((-1, gene_chunk))
    assert chunked.chunksize[1] != chunked.shape[1], (
        "rechunk did not produce a column-chunked array: "
        f"chunksize={chunked.chunksize}, shape={chunked.shape}"
    )
    return chunked


def read_reference(store: Path | str) -> AnnData:
    """Fully in-memory AnnData reference (X + layers as scipy CSC)."""
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
    np.testing.assert_array_equal(result.index, reference.index)
    assert_frame_equal(result, reference, atol=atol, check_dtype=False)


def list_gene_blocks(store: Path | str, key: str = "layers/counts") -> list[str]:
    """Sorted ``block_*`` names under ``key`` in a gene-block zarr store."""
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    names = sorted(n for n in parent.keys() if str(n).startswith("block_"))
    if not names:
        msg = f"no block_* subgroups under {key!r} in {store}"
        raise FileNotFoundError(msg)
    return names


def assert_on_disk_gene_blocks(
    store: Path | str, key: str = "layers/counts", *, min_blocks: int = 2
) -> list[str]:
    """Assert the store has multiple on-disk CSC gene-block subgroups."""
    names = list_gene_blocks(store, key)
    assert len(names) >= min_blocks, (
        f"expected >= {min_blocks} gene blocks under {key!r}, found {len(names)}: {names}"
    )
    return names


def load_geneblock_csc(store: Path | str, key: str = "layers/counts"):
    """Load gene-block CSC subgroups as one column-chunked CSC dask array.

    Each ``block_XXX`` is read with ``read_elem_lazy`` and concatenated on the gene
    axis so ``numblocks[1]`` matches the number of on-disk subgroups (unlike
    ``load_lazy_csc``, which rechunks a single CSC blob only in memory).
    """
    names = assert_on_disk_gene_blocks(store, key, min_blocks=2)
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    # read_elem_lazy may auto-split each CSC subgroup along genes (e.g. 1000);
    # rechunk so one dask column-chunk == one on-disk block_* subgroup.
    blocks = []
    for name in names:
        block = read_elem_lazy(parent[name])
        assert getattr(block._meta, "format", None) == "csc", (
            f"{key}/{name}: expected CSC dask meta, got {type(block._meta).__name__}"
        )
        blocks.append(block.rechunk((-1, block.shape[1])))
    combined = da.concatenate(blocks, axis=1)
    assert combined.numblocks[1] == len(names), (
        f"expected {len(names)} column blocks, got {combined.numblocks}"
    )
    assert combined.chunksize[1] != combined.shape[1], (
        "concatenated array is not column-chunked: "
        f"chunksize={combined.chunksize}, shape={combined.shape}"
    )
    return combined


def load_geneblock_csc_memory(store: Path | str, key: str = "layers/counts"):
    """Eager scipy CSC reference: hstack of the same on-disk gene blocks."""
    names = list_gene_blocks(store, key)
    group = zarr.open(str(store), mode="r")
    parent = _node(group, key)
    pieces = [csc_matrix(read_elem(parent[name])) for name in names]
    return hstack(pieces, format="csc")
