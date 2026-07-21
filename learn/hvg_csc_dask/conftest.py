"""Fixtures for the HVG CSC/dask tutorial suite.

Provides:
- ``hvg_store``: path to the generated zarr store (auto-built if missing);
- ``dask_client``: a real ``distributed`` ``LocalCluster`` + ``Client`` used by
  the "run under a real executor" scenario.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Importing _support prepends scanpy/src and anndata/src to sys.path.
import _support  # noqa: F401
import build_fixtures

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def hvg_store() -> Path:
    """Path to ``data/pbmc3k_hvg.zarr``, building it on demand.

    Auto-building keeps the suite runnable from a clean checkout (the store is
    git-ignored), while still reusing an existing store between runs.
    """
    return build_fixtures.build(force=False)


@pytest.fixture(scope="session")
def dask_client():
    """A real local dask scheduler for the executor scenario.

    ``processes=True`` + ``threads_per_worker=1`` avoids the numba-in-threads
    crash noted in scanpy's ``maybe_dask_process_context`` helper; each worker is
    its own process with a single thread.
    """
    distributed = pytest.importorskip("distributed")
    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()
