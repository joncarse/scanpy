"""Unit tests for `clip_square_sum` across array types and dask chunkings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.sparse as sps

from scanpy.preprocessing._highly_variable_genes._seurat_v3 import clip_square_sum
from testing.scanpy._pytest.marks import needs

if TYPE_CHECKING:
    from collections.abc import Callable


def _reference(x: np.ndarray, clip_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.minimum(x.astype(np.float64), clip_val[None, :])
    return np.square(clipped).sum(axis=0), clipped.sum(axis=0)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.poisson(2.0, size=(12, 8)).astype(np.float64)
    # positive clip values, as in the seurat_v3 use case (reg_std*sqrt(n)+mean)
    clip_val = rng.uniform(1.0, 4.0, size=8)
    return x, clip_val


def _as_dense(x: np.ndarray) -> np.ndarray:
    return x


def _as_csr(x: np.ndarray):
    return sps.csr_matrix(x)


def _as_csc(x: np.ndarray):
    return sps.csc_matrix(x)


def _as_dask_csr_row_chunked(x: np.ndarray):
    import dask.array as da

    return da.from_array(sps.csr_matrix(x), chunks=(5, -1))


def _as_dask_csc_col_chunked(x: np.ndarray):
    import dask.array as da

    return da.from_array(sps.csc_matrix(x), chunks=(-1, 3))


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(_as_dense, id="dense"),
        pytest.param(_as_csr, id="csr"),
        pytest.param(_as_csc, id="csc"),
        pytest.param(_as_dask_csr_row_chunked, id="dask_csr_row_chunked", marks=needs.dask),
        pytest.param(
            _as_dask_csc_col_chunked,
            id="dask_csc_col_chunked",
            marks=[
                needs.dask,
                pytest.mark.xfail(
                    reason="column-chunked clip_square_sum not implemented",
                    strict=True,
                ),
            ],
        ),
    ],
)
def test_clip_square_sum_matches_reference(
    data: tuple[np.ndarray, np.ndarray], builder: Callable
):
    x, clip_val = data
    exp_sq, exp_sum = _reference(x, clip_val)

    got_sq, got_sum = clip_square_sum(builder(x), clip_val)
    got_sq = np.asarray(got_sq)
    got_sum = np.asarray(got_sum)

    np.testing.assert_allclose(got_sq, exp_sq, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_sum, exp_sum, rtol=1e-6, atol=1e-6)
