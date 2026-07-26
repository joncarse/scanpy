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
    import distributed  # hard requirement for LocalCluster lessons (6/7)

    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=dask_client  where=client  pid=2237068  hits_at_site=1
    # topic: We are starting a local Dask cluster inside this pytest process.
    # --- lecture ---
    # We are starting a local Dask cluster inside this pytest process.
    #
    # Picture a small company: one office manager (the scheduler) and two
    # workers in separate rooms (separate OS processes). Your pytest script
    # is the client who walks into the lobby and hands the manager jobs.
    # The manager assigns jobs to workers; workers do the heavy lifting and
    # send results back.
    #
    # Why processes=True and threads_per_worker=1? Some of the numeric code
    # in this stack (notably numba-compiled kernels) is not happy when many
    # threads in one process all try to use it at once. Giving each worker
    # its own process and a single thread is the boring, reliable setup that
    # scanpy's own dask guidance recommends for this kind of work.
    #
    # dashboard_address=None turns off the fancy web UI so the test does not
    # fight over port 8787. We care about correctness and learning, not the
    # dashboard today.
    # --- facts at this step ---
    #   n_workers = 2
    #   threads_per_worker = 1
    #   processes = True
    #   pytest_pid = 2237068
    # --- locals / object fields at the call site ---
    #   distributed = <module 'distributed' from '/home/jonathan/scverse/.venv/lib/python3.14/site-packages/distributed/__init__.py'>
    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=dask_client  where=client  pid=2237068  hits_at_site=1
    # topic: The Client is connected.
    # --- lecture ---
    # The Client is connected. From this moment on, ordinary calls like
    # dask.array.compute(...) in this process will usually submit tasks to
    # THIS cluster instead of running a thread pool inside pytest alone.
    #
    # Look at the worker addresses and PIDs in the facts. Those PIDs are
    # different OS processes from the pytest PID. When we later pause a
    # breakpoint inside per_block during HVG, we are inside one of those
    # worker PIDs — not inside the pytest process that called
    # highly_variable_genes.
    #
    # The scheduler itself typically lives in the same process as the
    # client for LocalCluster. So "pytest process" ≈ client + scheduler
    # objects; "worker processes" ≈ the separate Python interpreters that
    # execute map_blocks tasks.
    # --- facts at this step ---
    #   client_pid = 2237068
    #   n_workers = 2
    #   worker_addresses = ['tcp://127.0.0.1:37387', 'tcp://127.0.0.1:45041']
    # --- locals / object fields at the call site ---
    #   distributed = <module 'distributed' from '/home/jonathan/scverse/.venv/lib/python3.14/site-packages/distributed/__init__.py'>
    #   cluster = LocalCluster(0b0b79e4, 'tcp://127.0.0.1:45913', workers=2, threads=2, memory=31.11 GiB)
    #   client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    # L7-LECTURE (real Lesson 7B run)
    # Step-by-step lecture note — inspected values from live execution.
    # function=dask_client  where=client  pid=2237068  hits_at_site=1
    # topic: Here are the operating-system process IDs of the workers.
    # --- lecture ---
    # Here are the operating-system process IDs of the workers. If you
    # attach a debugger "to a worker", you paste one of these PIDs into
    # the attach configuration.
    #
    # Remember: attaching to a worker lets you see gene-chunk tasks.
    # Stepping in the pytest process lets you see graph construction,
    # loess fitting, and da.compute waiting for results. You need both
    # perspectives to understand the full story.
    # --- facts at this step ---
    #   worker_os_pids = [2237350, 2237353]
    # --- locals / object fields at the call site ---
    #   distributed = <module 'distributed' from '/home/jonathan/scverse/.venv/lib/python3.14/site-packages/distributed/__init__.py'>
    #   cluster = LocalCluster(0b0b79e4, 'tcp://127.0.0.1:45913', workers=2, threads=2, memory=31.11 GiB)
    #   client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
    #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
    #   pids = [2237350, 2237353]
    try:
        yield client
    finally:
        # L7-LECTURE (real Lesson 7B run)
        # Step-by-step lecture note — inspected values from live execution.
        # function=dask_client  where=client  pid=2237068  hits_at_site=1
        # topic: Pytest has already decided pass/fail for the test body.
        # --- lecture ---
        # Pytest has already decided pass/fail for the test body. What you
        # are reading now is fixture teardown: we still must close the Client
        # and LocalCluster so worker processes exit and ports/memory are
        # released. Always clean up local clusters in fixtures so the next
        # test does not inherit a half-dead cluster.
        # --- locals / object fields at the call site ---
        #   distributed = <module 'distributed' from '/home/jonathan/scverse/.venv/lib/python3.14/site-packages/distributed/__init__.py'>
        #   cluster = LocalCluster(0b0b79e4, 'tcp://127.0.0.1:45913', workers=2, threads=2, memory=31.11 GiB)
        #   client = <Client: 'tcp://127.0.0.1:45913' processes=2 threads=2, memory=31.11 GiB>
        #   workers = {'tcp://127.0.0.1:37387': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {}, 'local_directory': '/tmp/dask-scratch-space/worker-sbf00p7t', 'na...
        #   pids = [2237350, 2237353]
        client.close()
        cluster.close()
