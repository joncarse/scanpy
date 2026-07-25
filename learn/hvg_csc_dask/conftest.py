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
from _support import narrate

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
    import os

    import distributed  # hard requirement for LocalCluster lessons (6/7)

    narrate(
        "client",
        "dask_client fixture: creating LocalCluster",
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        pytest_pid=os.getpid(),
    )
    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    workers = client.scheduler_info().get("workers", {})
    narrate(
        "client",
        "dask_client fixture: cluster ready",
        client_pid=os.getpid(),
        n_workers=len(workers),
        worker_pids=[w.get("id") for w in workers.values()],
        worker_addresses=list(workers),
    )
    # Prefer OS PIDs when available on worker info / cluster.workers.
    try:
        pids = [cluster.workers[k].pid for k in sorted(cluster.workers)]
        narrate("client", "dask_client fixture: worker OS PIDs", worker_os_pids=pids)
    except Exception:  # noqa: BLE001 — narration only
        pass
    try:
        yield client
    finally:
        narrate("client", "dask_client fixture: shutting down client/cluster")
        client.close()
        cluster.close()
