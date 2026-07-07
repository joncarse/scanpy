# HVG on column-chunked CSC dask: a guided debugging course

A graduated set of lessons for understanding how `scanpy.pp.highly_variable_genes`
(HVG) works on **column-chunked CSC dask arrays**, using the on-disk `pbmc3k`
fixtures in this folder. Each lesson has a matching entry in
`[.vscode/launch.json](../../.vscode/launch.json)` (group "HVG lessons"), so you can
start it under the debugger with one click.

Work through them in order: they go from the simplest dispersion flavor to a full
pipeline and finally a real distributed executor.

## Before you start

1. **Use the repo venv.** Opening this fork in its own Cursor/VS Code window should
   pick up `[.venv/bin/python](../../.venv/bin/python)` (Python 3.14.6) via
   `[.vscode/settings.json](../../.vscode/settings.json)`. Create it once from the
   repo root if missing:

   ```bash
   python3.14 -m venv .venv
   . .venv/bin/activate
   pip install -e .
   pip install -r learn/hvg_csc_dask/requirements.txt
   ```

   The launch configs use `${command:python.interpreterPath}` (that venv). The suite
   also prepends this fork's `src/` to `sys.path`, so `import scanpy` always
   resolves to the code you are stepping through.
2. **Build the fixtures once** (Lesson 0). Everything else auto-builds them on
   first use, but running Lesson 0 explicitly is the best way to see what is on
   disk.
3. **`justMyCode` is already `false`** in every lesson config, so you can step from
   the test straight into scanpy's source.

Key source files you will land in:

- `[src/scanpy/preprocessing/_highly_variable_genes/_main.py](../../src/scanpy/preprocessing/_highly_variable_genes/_main.py)` - the dispatcher.
- `[src/scanpy/preprocessing/_highly_variable_genes/_dispersion.py](../../src/scanpy/preprocessing/_highly_variable_genes/_dispersion.py)` - `seurat`/`cell_ranger`.
- `[src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py](../../src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py)` - `seurat_v3`/`seurat_v3_paper` and the new dask code.
- `[src/scanpy/get/_aggregated.py](../../src/scanpy/get/_aggregated.py)` - per-batch `aggregate`.
- `[src/scanpy/preprocessing/_normalization.py](../../src/scanpy/preprocessing/_normalization.py)` - `normalize_total`.

---

## Lesson 0 - Build & inspect the fixtures

**Launch:** `Lesson 0: Build & inspect fixtures` (runs `build_fixtures.py --force`).

**Goal:** understand the difference between *on-disk sparse storage* and *dask chunk
layout* - the whole premise of this suite.

**Breakpoints:**

- `build_fixtures.py::build`, at `out.write_zarr(STORE)`. Inspect `out`: `X` and
  `layers["counts"]` are both `csc_matrix`, `obs["batch"]` is categorical.
- `_support.py::load_lazy_csc`, after `lazy = read_elem_lazy(...)` and again after
  `.rechunk((-1, gene_chunk))`.

**Observe / ask yourself:**

- On disk the store holds a plain CSC component. Where does "column-chunked" come
  from? (Answer: only at read time, via `.rechunk((-1, gene_chunk))`.)
- Before rechunk, `lazy.chunks` is a single block; after, `chunks[1]` is
  `(200, 200, 164)` and `chunksize[1] != shape[1]`. That inequality is exactly what
  the new HVG code keys off.
- `X` is log-normalized (for `seurat`/`cell_ranger`); `layers["counts"]` is raw
  counts (for `seurat_v3*`). Why do the flavors need different inputs?

---

## Lesson 1 - Dispersion flavor, single batch (`seurat`)

**Launch:** `Lesson 1: Dispersion (seurat, single batch)`
(`test_dispersion_matches_in_memory[single-seurat]`).

**Goal:** trace the simplest HVG path end to end on a lazy CSC `X`.

**Breakpoints (set in this order, then step):**

1. `_main.py::highly_variable_genes` - the dispatcher. Watch `flavor` route away
   from the `seurat_v3` branch into `_highly_variable_genes_single_batch`.
2. `_dispersion.py::_highly_variable_genes_single_batch`:
   - the `np.expm1(X)` step (only for `flavor="seurat"`; step here and note it is
     skipped for `cell_ranger` in Lesson 2);
   - `stats.mean_var(X, axis=0)` - this is the dask reduction. Step in and see it
     run per gene-block because the array is column-chunked.
   - `_get_mean_bins` then `_get_disp_stats` - binning + per-bin z-scoring of
     dispersion.
   - `_subset_genes` / `_nth_highest` - the top-N selection.

**Observe / ask yourself:**

- `mean_var` returns dask arrays here; where do they get materialized to numpy?
- The test selects `n_top_genes=n_vars` (all genes). Why? (So the assertion checks
  the numeric per-gene statistics rather than a top-N boundary that can flip on
  sub-`atol` floating-point differences.)

---

## Lesson 2 - Dispersion flavor, batched (`cell_ranger`)

**Launch:** `Lesson 2: Dispersion (cell_ranger, batched)`
(`test_dispersion_matches_in_memory[batched-cell_ranger]`).

**Goal:** see how batching wraps the single-batch path, and how `cell_ranger`
differs from `seurat`.

**Breakpoints:**

- `_dispersion.py::_highly_variable_genes_batched` - note it loops per batch and
  calls the single-batch logic for each, then combines.
- `_per_batch_func` - the per-batch dispatch helper.
- Back in `_highly_variable_genes_single_batch`, confirm the `expm1` step is
  **not** taken for `cell_ranger`.

**Observe / ask yourself:**

- How is the per-batch subset taken from a column-chunked dask array (row masking on
  the observation axis while the gene axis stays chunked)?
- What column does batched mode add that single-batch does not?

---

## Lesson 3 - `seurat_v3`, single batch (the new dask code)

**Launch:** `Lesson 3: seurat_v3 (single batch, new clip branch)`
(`test_seurat_v3_matches_in_memory[single-seurat_v3]`).

**Goal:** this is the heart of the feature. `seurat_v3` consumes raw counts from
`layers["counts"]` and exercises the code we added for column-chunked CSC dask.

**Breakpoints:**

1. `_seurat_v3.py::_raise_if_unsupported_dask_chunking` - the relaxed guard. Step
   through and see why column-chunked **CSC** with an unchunked observation axis is
   allowed while other feature-chunked layouts are rejected.
2. `_seurat_v3.py::_highly_variable_genes_seurat_v3`:
   - `stats.mean_var(data, axis=0)`;
   - the `loess(x, y, ...).fit()` regression of variance on mean;
   - `clip_square_sum(data_batch, clip_val)`.
3. `_seurat_v3.py::clip_square_sum` (the `DaskArray` registration) - watch it detect
   `data_batch.chunksize[1] != data_batch.shape[1]` and branch into...
4. `_seurat_v3.py::_clip_square_sum_feature_chunked` - **the new path**. Note how
   each block slices `clip_val[col_subset]` using `block_info` and the per-block
   clipped sums are concatenated along the gene axis (not summed across blocks, as
   the row-chunked path does).

**Observe / ask yourself:**

- Why can feature-chunked blocks be concatenated rather than reduced? (Each block
  holds *all* observations for a disjoint set of genes, so its per-gene sums are
  already final.)
- Compare with the row-chunked branch just above it in `clip_square_sum`: what would
  break if you fed a row-chunked array into `_clip_square_sum_feature_chunked`?

---

## Lesson 4 - `seurat_v3`, batched (aggregate + row masking)

**Launch:** `Lesson 4: seurat_v3 (batched, aggregate + row masking)`
(`test_seurat_v3_matches_in_memory[batched-seurat_v3]`).

**Goal:** the most involved path: per-batch mean/var via `scanpy.get.aggregate` on a
column-chunked dask array, plus row masking per batch.

**Breakpoints:**

- `_seurat_v3.py::_highly_variable_genes_seurat_v3`, at the
  `aggregate(adata_agg, by="__hvg_v3_batch_info__", func=["mean", "var"])` call.
- `[src/scanpy/get/_aggregated.py](../../src/scanpy/get/_aggregated.py)` `aggregate_dask` - step into the
  column-chunked branch and watch it aggregate per gene-block.
- Back in `_highly_variable_genes_seurat_v3`, the `data_batch = data[batch_info == b]`
  line - row masking on a column-chunked array.

**Observe / ask yourself:**

- How does `aggregate` keep the gene-axis chunking intact while grouping over
  observations?
- `seurat_v3` vs `seurat_v3_paper` differ only in sort order of the final ranking -
  find the `sort_cols` / `sort_ascending` branch and explain the difference.

---

## Lesson 5 - Full pipeline (`normalize_total` layout flip)

**Launch:** `Lesson 5: Full pipeline (normalize_total layout flip)`
(`test_full_pipeline_matches_in_memory`).

**Goal:** run `normalize_total` -> `log1p` -> HVG from lazy raw counts and see a
subtle, important detail: the sparse *layout* can change mid-pipeline.

**Breakpoints:**

- `[src/scanpy/preprocessing/_normalization.py](../../src/scanpy/preprocessing/_normalization.py)`, around line 257:

  ```python
  if isinstance(x, CSCBase):
      x = x.tocsr()
  ```

- In the test, the two assertions after `normalize_total`: the in-memory matrix is
  now `csr_matrix`, but the dask array keeps its CSC `_meta` and column chunking.

**Observe / ask yourself:**

- Why does the in-memory path flip CSC -> CSR while the dask path does not?
- Despite the layout difference, the final HVG statistics still match. What does that
  tell you about what HVG actually depends on?

---

## Lesson 6 - Real dask executor (`LocalCluster`)

**Launch:** `Lesson 6: Real dask executor (LocalCluster)`
(`test_seurat_v3_under_local_cluster`).

**Goal:** the same `seurat_v3` column-chunked computation, but executed on a real
`distributed.LocalCluster` (2 worker processes), not the default synchronous
scheduler.

**Breakpoints:**

- `conftest.py::dask_client` - see the cluster/client come up
  (`processes=True`, `threads_per_worker=1`).
- In the test, the `with distributed.get_task_stream(client) as ts:` block and the
  `assert len(ts.data) > 0` afterwards.

**Observe / ask yourself:**

- Why `processes=True` and one thread per worker? (The per-block `clip_square_sum`
  calls into a numba `njit` kernel; running it under multiple threads per worker can
  crash - scanpy guards this in `maybe_dask_process_context`.)
- Breakpoints inside `clip_square_sum` may **not** hit the way they do in Lesson 3,
  because the work now runs in separate worker processes. This is the key mental
  model shift: with a distributed scheduler, your per-block functions execute
  elsewhere. Compare the experience to Lesson 3's in-process scheduler.
- `ts.data` is the captured task stream; a non-empty list proves work actually ran on
  the cluster.

---

## Suggested progression recap

| Lesson | Launch config                                    | Focus                                             |
| ------ | ------------------------------------------------ | ------------------------------------------------- |
| 0      | Build & inspect fixtures                         | on-disk CSC vs dask column chunking               |
| 1      | Dispersion (seurat, single batch)                | simplest full HVG path                            |
| 2      | Dispersion (cell_ranger, batched)                | batching wrapper; flavor differences              |
| 3      | seurat_v3 (single batch, new clip branch)        | the new `_clip_square_sum_feature_chunked` path   |
| 4      | seurat_v3 (batched, aggregate + row masking)     | `aggregate_dask` + per-batch masking              |
| 5      | Full pipeline (normalize_total layout flip)      | CSC -> CSR in memory vs CSC-preserving dask        |
| 6      | Real dask executor (LocalCluster)                | distributed execution; where blocks actually run  |
