# TODO

Known gaps, deferred parity items, and follow-ups from the 2026-07 parity
campaign (reference: `_tinygrad` @ 7eb197b1b). Maintainer notes — reference
anchors point at the tinygrad clone.

## Open bugs

- **CPU core-count divergence** (`upcast_lane_load`/`upcast_lane_store`
  stage5_cpu): tolk derives 8 threads where the reference derives 12 on the
  same host (`core_id` range (0,7) vs (0,11), shard strides scale
  accordingly). Pre-dates the campaign. Find where the reference sources its
  count (`os.cpu_count()` vs performance cores) and mirror it in the CPU
  threading path (gpudims/runtime).
- **Param shape emission divergence** (systemic, ~32 parity cases):
  shapeless params emit a `NOOP` shape child where the reference dumps
  expect `-1` (elementwise_add, reduce_rows, multi_param, multi_output,
  sum_reduce, matmul_small, …). Cases building shaped params
  (contiguous_add, symbolic_shrink) pass, so the divergence is in the
  default/sentinel shape path of param construction or its printing.
- **test_schedule_rangeify reduce_axis failures** (4): "rewrite cycle
  detected" and direct-source indexing failures on reduce_axis-involved
  cases (including symbolic-shrink-then-reduce_axis). Reduce-payload
  churn suspected; needs an owner with the suite runnable.

## Deferred parity divergences (each with a known blocker)

- **Bare-view aliasing**: contiguous folds materialize a fresh buffer + copy;
  the reference returns a buffer-identity view (callify.py
  `contiguous_mops_to_view`). Runtime cost: one allocation + copy kernel per
  occurrence per execution (replays every call under jit) plus double
  residency. Blocker: the tag side-table re-tags rewrite results
  (`propagate_tags` on_rebuild) and re-wraps the bare view into a
  materialized copy. Needs the tag-machinery rework plus an end-to-end
  realize test asserting `x[a:b].realize()` aliases its source.
- **BITCAST/COPY fold extension** (callify): same tag blocker; the COPY arm
  additionally needs the disk-copy push rules tolk does not port.
- **expand_bitcast** (rangeify): differing-itemsize bitcast reshaping
  unported; needs movement-composition helpers (usum/squeeze/flatten-style).
  Only fires for quant/disk-style bitcasts; no golden exercises it.
- **Host I/O via allocator bridge** (`frontend/run.ml`): upload/download use
  `Buffer.copyin`/`as_bytes` directly because tolk has no host pseudo-device
  to route a `copy_from` through (reference: device.py:113-115 seeds via a
  PYTHON-device buffer). Converges to `copy_from` if a host device lands.
- **Image coordinate form**: the gater/coalese/renderer accept both
  two-axis `INDEX(buf, y, x)` (reference form) and stacked
  `INDEX(buf, STACK[y,x])`. Converge on two-axis and drop the stacked branch,
  gated on an image golden generated from the clone (none exists — the image
  path is uncovered by committed goldens).
- **keccak/hash** (frontend): the reference moved SHA3/keccak into the op
  mixin; tolk does not port it.

## Design debt

- **nativeint dispatch waist**: `Allocator.addr`/`prog.call`/`Graph.node`
  flatten buffers to `nativeint`, forcing the Metal token registry
  (`tolk_metal.ml`) and an `Obj.magic` in `Buffer.transfer`, and making
  multi-GPU CUDA transfer inexpressible (no device identities on the
  transfer). Plan: thread opaque `Buffer.t` (or a backend-downcast handle)
  through dispatch; runtimes bind (object, offset) themselves. Interim hard
  contract (enforced): the engine roots offset-view buffers across launch —
  the Metal registry is weak and a GC'd view raises at token resolve.
- **`uf` scalar-promotion helper triplicated** across
  `elementwise.ml`/`op.ml`/`rand.ml` to avoid public surface; fold into a
  shared non-public home if dune allows one.
- **`Symbolic.index_pushing` export**: likely consumer-less since the
  simplifier composes plain `symbolic` (which carries movement cleanup);
  delete the export if a consumer sweep confirms.
- **Upat `Prefix` source pattern**: redundant with `Fixed` + `allow_any_len`,
  unused in lib; DSL simplification candidate.
- **Upat `alu` operand dtype**: patterns don't constrain operand dtypes
  (benign — no rule uses typed operands; becomes a bug if one appears).

## Performance follow-ups

- **Weak memo caches**: the uop layer's `Ref_tbl` memo caches (device,
  addrspace, min_max, shape, axis, ranges, child_ops) and side_metadata are
  non-weak, so entries outlive collected nodes — unbounded growth over long
  processes. Fix with Ephemeron-style weak tables; benchmark the hot path
  before/after.
- **div/mod fold memoization**: `fold_divmod_general` recomputes on shared
  sub-expressions; the reference caches (`@functools.cache`). A `Ref_tbl`
  memo bounds pathological blowup; only worth it if profiling shows cost.
- **pm_replace_buf global gate** (callify): the reference gates buffer→param
  replacement on GLOBAL address space; tolk replaces all. Benign today (only
  global output buffers reach the assigns).

## Missing tests

- End-to-end realize aliasing test (gates the bare-view fix).
- `multi_stack` parity case (the STACK sharding rule has no coverage).
- Image golden from the clone (gates the image-coordinate convergence).
- Golden asserting the group-reduce gate-mask boolean shape
  (`Cmpeq` vs the reference's compare-nest — currently proven only
  indirectly by gated load/store goldens).
- BEAM-on-CPU test (would have caught the wait-timing bug where every
  candidate tied at infinity).
- `Buffer.copy_from` contract test (fail-loud before the runner is
  installed; delegation after) — needs a new `test_device.ml`.
- WMMA estimate unit test (flops factor covered only by goldens).
- Re-verify `bf16_vector_load_reindexes_shrink` (decomp) end-to-end: it
  relies on the LOAD inheriting the shrink's width-2 shape.
- Two migrated image-test assertions to value-check once suites run:
  invalid-index sentinels under `pm_remove_invalid`, and
  `pm_move_where_on_load` behavior after the pointer-flag removal.

## Misc

- `Uop.program_vals` raises bare `Not_found` for an unbound variable; the
  reference raises a descriptive error naming the variable and its user.
- `pm_clean_up_group_sink` exists as a codegen-local matcher; if the
  symbolic layer ever exports it, dedupe the local copy.
- OpenCL `aux` metadata may need the buffer shape for `clCreateImage` if an
  OpenCL runtime lands; verify against reference output first.
- Dormant: a `weakfloat` store value into a float32 buffer is not
  concretized in a degenerate loop-invariant store (realistic programs store
  already-concrete values).
