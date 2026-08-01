"""Edge-case and rejection tests for column-chunked CSC dask HVG."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.sparse as sps
from pandas.testing import assert_frame_equal, assert_index_equal

import scanpy as sc
from testing.scanpy._helpers.data import pbmc3k
from testing.scanpy._pytest.marks import needs
from testing.scanpy._pytest.params import ARRAY_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData

CSC_DASK_PARAMS = [p for p in ARRAY_TYPES if "1d_chunked" in p.id and "csc" in p.id]
SEURAT_V3_FLAVORS = ["seurat_v3", "seurat_v3_paper"]
UNSUPPORTED_MSG = r"Only dask arrays with chunking along the first axis are supported"


def _counts_adata(*, batched: bool = False) -> AnnData:
    adata = pbmc3k()[:1500, :1000].copy()
    adata.X = np.abs(adata.X.toarray()).astype(np.float32)
    if batched:
        adata.obs["batch"] = np.tile(["a", "b"], adata.shape[0] // 2).astype(str)
    return adata


def _as_dense_feature_chunked(x: np.ndarray):
    import dask.array as da

    return da.from_array(x, chunks=(400, 200))


def _as_csr_feature_chunked(x: np.ndarray):
    import dask.array as da

    return da.from_array(sps.csr_matrix(x), chunks=(-1, 200))


def _as_csc_dual_chunked(x: np.ndarray):
    import dask.array as da

    return da.from_array(sps.csc_matrix(x), chunks=(400, 200))


@needs.dask
@needs.skmisc
@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(_as_dense_feature_chunked, id="dense_feature_chunked"),
        pytest.param(_as_csr_feature_chunked, id="csr_feature_chunked"),
        pytest.param(_as_csc_dual_chunked, id="csc_obs_and_feature_chunked"),
    ],
)
@pytest.mark.parametrize("flavor", SEURAT_V3_FLAVORS)
def test_seurat_v3_rejects_unsupported_chunkings(builder: Callable, flavor: str):
    """Column-chunked CSC is allowed; dense/CSR feature-chunked and dual-axis CSC are not."""
    adata = _counts_adata()
    adata.X = builder(np.asarray(adata.X))
    with pytest.raises(ValueError, match=UNSUPPORTED_MSG):
        sc.pp.highly_variable_genes(
            adata, flavor=flavor, n_top_genes=50, inplace=False
        )


@needs.dask
@needs.skmisc
@pytest.mark.parametrize("flavor", SEURAT_V3_FLAVORS)
@pytest.mark.parametrize("to_dask", CSC_DASK_PARAMS)
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
def test_seurat_v3_csc_dask_layer_matches_memory(
    flavor: str, to_dask: Callable, batched: bool
):
    """``layer=`` path must use the same feature-chunked CSC logic as ``X``."""
    base = _counts_adata(batched=batched)
    dense_x = np.asarray(base.X)

    adata_mem = base.copy()
    adata_mem.layers["counts"] = sps.csc_matrix(dense_x)

    adata_dask = base.copy()
    adata_dask.layers["counts"] = to_dask(dense_x)

    kwargs: dict = dict(
        flavor=flavor,
        layer="counts",
        n_top_genes=50,
        inplace=False,
    )
    if batched:
        kwargs["batch_key"] = "batch"

    out_mem = sc.pp.highly_variable_genes(adata_mem, **kwargs)
    out_dask = sc.pp.highly_variable_genes(adata_dask, **kwargs)

    assert_index_equal(out_mem.index, out_dask.index)
    assert_frame_equal(out_mem, out_dask, atol=1e-4, check_dtype=False)


@needs.dask
@needs.skmisc
@pytest.mark.parametrize("flavor", SEURAT_V3_FLAVORS)
@pytest.mark.parametrize("to_dask", CSC_DASK_PARAMS)
def test_seurat_v3_csc_dask_n_top_genes_selection(flavor: str, to_dask: Callable):
    """Top-N HVG membership and ranks agree for column-chunked CSC dask."""
    base = _counts_adata()
    dense_x = np.asarray(base.X)
    n_top = 75

    adata_mem = base.copy()
    adata_mem.X = sps.csc_matrix(dense_x)
    adata_dask = base.copy()
    adata_dask.X = to_dask(dense_x)

    out_mem = sc.pp.highly_variable_genes(
        adata_mem, flavor=flavor, n_top_genes=n_top, inplace=False
    )
    out_dask = sc.pp.highly_variable_genes(
        adata_dask, flavor=flavor, n_top_genes=n_top, inplace=False
    )

    assert out_mem["highly_variable"].sum() == n_top
    assert out_dask["highly_variable"].sum() == n_top
    assert_frame_equal(out_mem, out_dask, atol=1e-4, check_dtype=False)


@needs.dask
@needs.skmisc
@pytest.mark.parametrize("flavor", SEURAT_V3_FLAVORS)
def test_seurat_v3_csc_dask_uneven_gene_chunks(flavor: str):
    """Explicit uneven gene-chunk sizes still match in-memory CSC."""
    import dask.array as da

    base = _counts_adata()
    dense_x = np.asarray(base.X)
    # force a short final gene block (1000 genes → 400 + 400 + 200)
    x_dask = da.from_array(sps.csc_matrix(dense_x), chunks=(-1, (400, 400, 200)))
    assert x_dask.chunks[1][-1] != x_dask.chunks[1][0]

    adata_mem = base.copy()
    adata_mem.X = sps.csc_matrix(dense_x)
    adata_dask = base.copy()
    adata_dask.X = x_dask

    out_mem = sc.pp.highly_variable_genes(
        adata_mem, flavor=flavor, n_top_genes=base.n_vars, inplace=False
    )
    out_dask = sc.pp.highly_variable_genes(
        adata_dask, flavor=flavor, n_top_genes=base.n_vars, inplace=False
    )
    assert_frame_equal(out_mem, out_dask, atol=1e-4, check_dtype=False)


@needs.zarr
@needs.dask
@needs.skmisc
@pytest.mark.parametrize("flavor", SEURAT_V3_FLAVORS)
def test_seurat_v3_zarr_backed_csc_dask(tmp_path: Path, flavor: str):
    """On-disk CSC zarr + gene-axis rechunk matches in-memory column-chunked CSC dask."""
    import dask.array as da
    import zarr
    from anndata.experimental import read_elem_lazy
    from anndata.io import write_elem

    from scanpy._compat import CSCBase, DaskArray

    base = _counts_adata()
    x_csc = sps.csc_matrix(np.asarray(base.X))

    store = str(tmp_path / "hvg_seurat_v3.zarr")
    group = zarr.open_group(store, mode="w")
    write_elem(group, "X", x_csc)

    lazy_x = read_elem_lazy(zarr.open_group(store, mode="r")["X"])
    assert isinstance(lazy_x, DaskArray)
    assert isinstance(lazy_x._meta, CSCBase)
    lazy_x = lazy_x.rechunk((-1, 200))
    assert lazy_x.chunksize[1] != lazy_x.shape[1]

    mem_x = da.from_array(x_csc, chunks=(-1, 200))

    adata_mem = base.copy()
    adata_mem.X = mem_x
    adata_zarr = base.copy()
    adata_zarr.X = lazy_x

    out_mem, out_zarr = (
        sc.pp.highly_variable_genes(ad, flavor=flavor, n_top_genes=50, inplace=False)
        for ad in [adata_mem, adata_zarr]
    )
    assert_index_equal(base.var_names, out_zarr.index, check_names=False)
    assert_frame_equal(out_mem, out_zarr, atol=1e-4, check_dtype=False)
