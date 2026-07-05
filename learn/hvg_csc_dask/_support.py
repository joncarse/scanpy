"""Shared helpers for the HVG CSC/dask tutorial suite.

Kept separate from ``conftest.py`` so both the fixtures and the individual test
modules can import the same loaders/comparators.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Resolve `import scanpy` to THIS fork checkout: scanpy/learn/hvg_csc_dask -> scanpy/src
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import zarr  # noqa: E402
from anndata import read_zarr  # noqa: E402
from anndata.experimental import read_elem_lazy  # noqa: E402
from pandas.testing import assert_frame_equal  # noqa: E402

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
