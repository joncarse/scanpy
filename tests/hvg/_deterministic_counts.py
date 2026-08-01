"""Deterministic count matrices for the HVG memory/ooc test matrix.

Expression at cell ``i``, gene ``j`` is a pure function of the indices so every
run (memory CSC or mocked lazy CSC-dask) sees identical nonnegative integer
UMIs. Many entries are zero so the sparse/CSC path stays realistic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sps
from anndata import AnnData

import scanpy as sc

if TYPE_CHECKING:
    from collections.abc import Callable

    from scanpy._compat import DaskArray

N_OBS = 1000
N_VARS = 1000
GENE_CHUNK = 250
SEURAT_V3_FLAVORS = frozenset({"seurat_v3", "seurat_v3_paper"})

# Nonzero when (i * 31 + j * 17) % SPARSITY_MOD == 0 → ~1/7 fill.
SPARSITY_MOD = 7


def umi(i: int, j: int) -> int:
    """Return the UMI count for cell ``i`` and gene ``j``."""
    if (i * 31 + j * 17) % SPARSITY_MOD != 0:
        return 0
    # Bounded positive integer depending only on indices.
    return 1 + (i * 13 + j * 7) % 50


def deterministic_dense(
    n_obs: int = N_OBS, n_vars: int = N_VARS
) -> np.ndarray:
    """Build a dense float32 count matrix from :func:`umi` (vectorized)."""
    i = np.arange(n_obs, dtype=np.int64)[:, None]
    j = np.arange(n_vars, dtype=np.int64)[None, :]
    mask = (i * 31 + j * 17) % SPARSITY_MOD == 0
    values = (1 + (i * 13 + j * 7) % 50).astype(np.float32)
    return np.where(mask, values, np.float32(0))


def deterministic_csc(
    n_obs: int = N_OBS, n_vars: int = N_VARS
) -> sps.csc_matrix:
    """In-memory CSC view of the deterministic counts."""
    return sps.csc_matrix(deterministic_dense(n_obs, n_vars))


def deterministic_csc_dask(
    n_obs: int = N_OBS,
    n_vars: int = N_VARS,
    *,
    gene_chunk: int = GENE_CHUNK,
) -> DaskArray:
    """Column-chunked CSC dask array (virtual gene chunks, single CSC payload)."""
    import dask.array as da

    x = da.from_array(deterministic_csc(n_obs, n_vars), chunks=(-1, gene_chunk))
    assert x.chunksize[0] == n_obs
    assert x.chunksize[1] != n_vars or n_vars <= gene_chunk
    return x


def _obs_names(n_obs: int) -> list[str]:
    return [f"cell_{i:04d}" for i in range(n_obs)]


def _var_names(n_vars: int) -> list[str]:
    return [f"gene_{j:04d}" for j in range(n_vars)]


def make_counts_adata(
    *,
    n_obs: int = N_OBS,
    n_vars: int = N_VARS,
    batched: bool = False,
    x_builder: Callable[[int, int], object] | None = None,
) -> AnnData:
    """AnnData with deterministic raw counts in ``X`` and optional batch labels."""
    builder = x_builder or deterministic_csc
    adata = AnnData(X=builder(n_obs, n_vars))
    adata.obs_names = _obs_names(n_obs)
    adata.var_names = _var_names(n_vars)
    if batched:
        adata.obs["batch"] = np.array(
            ["a"] * (n_obs // 2) + ["b"] * (n_obs - n_obs // 2)
        )
        assert set(adata.obs["batch"]) == {"a", "b"}
    return adata


def prepare_for_flavor(adata: AnnData, flavor: str) -> AnnData:
    """Copy and preprocess ``adata`` for the given HVG flavor."""
    out = adata.copy()
    if flavor not in SEURAT_V3_FLAVORS:
        sc.pp.normalize_total(out, target_sum=1e4)
        sc.pp.log1p(out)
    return out


Storage = Literal["memory", "ooc"]


def make_hvg_adata(
    *,
    flavor: str,
    storage: Storage,
    batched: bool,
    n_obs: int = N_OBS,
    n_vars: int = N_VARS,
    gene_chunk: int = GENE_CHUNK,
) -> AnnData:
    """Build a flavor-ready AnnData for ``memory`` or mocked ``ooc`` storage.

    For ``ooc``, call ``anndata.experimental.read_elem_lazy`` (tests monkeypatch
    that symbol) then rechunk along genes — same seam as a zarr-backed load.
    """
    if storage == "memory":
        adata = make_counts_adata(
            n_obs=n_obs, n_vars=n_vars, batched=batched, x_builder=deterministic_csc
        )
        return prepare_for_flavor(adata, flavor)

    from anndata.experimental import read_elem_lazy

    # Production-shaped load: lazy read then gene-axis rechunk.
    # The elem argument is ignored by the test mock.
    lazy_x = read_elem_lazy(object())
    lazy_x = lazy_x.rechunk((-1, gene_chunk))
    adata = AnnData(X=lazy_x)
    adata.obs_names = _obs_names(n_obs)
    adata.var_names = _var_names(n_vars)
    if batched:
        adata.obs["batch"] = np.array(
            ["a"] * (n_obs // 2) + ["b"] * (n_obs - n_obs // 2)
        )
    return prepare_for_flavor(adata, flavor)
