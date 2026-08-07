# TODO

Known gaps, deferred parity items, and follow-ups from the 2026-07 parity
campaign (reference: `_tinygrad` @ 7eb197b1b). Maintainer notes — reference
anchors point at the tinygrad clone.

## Open bugs

- **Renderer vector stores through scalar pointers**: full-model codegen can
  retain a vectorized value at a scalar store site. The float16 GPT-2
  loss-multiply backward renders `make_float50257(...)` on CUDA, which NVRTC
  rejects; a pmapped dropout backward renders a 3-wide float store through a
  scalar pointer on CPU. The repros are recorded in Kaun's GPT-2 `train.ml`;
  both likely share an owner in the devectorize/store path.
- **Pure-constant materialization**: `Run.buffer_of` raises when a tensor's graph
  folds to a constant expression with no storage instead of materializing it.
  The dropout `p=1` test currently works around this.

## Deferred parity divergences (each with a known blocker)

- **Leading-dim view folds**: last-axis contiguous views fold to schedule-free
  aliases (matching the reference), but callify's local
  `contiguous_view_offset`/`make_slice` bails when a leading dim is not kept
  in full, so e.g. a 2-D row slice still materializes a copy. The exported
  `Uop.contiguous_view_offset` already handles leading dims — align the fold
  to it and delete the partial local reimplementation. Schedule-shape change
  (kernel counts drop), so verify against the golden suites.
- **BITCAST/COPY fold extension** (callify): same tag blocker; the COPY arm
  additionally needs the disk-copy push rules tolk does not port.
- **expand_bitcast** (rangeify): differing-itemsize bitcast reshaping
  unported; needs movement-composition helpers (usum/squeeze/flatten-style).
  It blocks half-precision `Rand.rand`; no golden exercises it.
- **Host I/O via allocator bridge** (`frontend/run.ml`): upload/download use
  `Buffer.copyin`/`as_bytes` directly because tolk has no host pseudo-device
  to route a `copy_from` through (reference: device.py:113-115 seeds via a
  PYTHON-device buffer). Converges to `copy_from` if a host device lands.
- **Image coordinate form**: the gater/coalese/renderer accept both
  two-axis `INDEX(buf, y, x)` (reference form) and stacked
  `INDEX(buf, STACK[y,x])`. Converge on two-axis and drop the stacked branch,
  gated on an image golden generated from the clone (none exists — the image
  path is uncovered by committed goldens).

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
- **Upat `Prefix` source pattern**: redundant with `Fixed` + `allow_any_len`,
  unused in lib; DSL simplification candidate.

## Multi-device follow-ups

- **Frontend sharding API**: add `Tensor.shard`/`shard_`/`to_` for device tuples.
  Scheduling and execution underneath are complete and golden-pinned; move
  `shard`/`unshard` to uop ownership.
- **CUDA peer access** (needs at least two GPUs): call
  `cuDeviceCanAccessPeer`/`cuCtxEnablePeerAccess` when opening devices.
- **Eventful cross-context transfer** (needs at least two GPUs): give allocator
  `transfer` source and destination device parameters. This touches every
  backend record and depends on resolving the nativeint dispatch waist above.
- **Multi-device graph batching**: build per-shard graph nodes under the
  `MultiGraphRunner` same-backend rule. Multi calls currently run correctly but
  ungraphed and unbatched.

## Performance follow-ups

- **CPU matmul runtime trails the reference ~25-30%** at N=512/1024 (Track 2
  indicative bench, `bench/runtime/`: 71.8 vs 90.7 and 68.3 vs 98.3 GFLOP/s).
  Kernels are byte-identical at 128³ (compare suite), so the gap at larger
  shapes most likely comes from optimizer heuristic selection (tile/upcast
  choices), not codegen. Runtime divergence is out of scope by design —
  pick this up only if the runtime track opens.
- **Weak memo caches**: the uop layer's `Ref_tbl` memo caches (device,
  addrspace, min_max, shape, axis, ranges, child_ops) and side_metadata are
  non-weak, so entries outlive collected nodes — unbounded growth over long
  processes. Fix with Ephemeron-style weak tables; benchmark the hot path
  before/after.

## Missing tests

- `multi_stack` parity case (the STACK sharding rule has no coverage).
- Image golden from the clone (gates the image-coordinate convergence).
- Golden asserting the group-reduce gate-mask boolean shape
  (`Cmpeq` vs the reference's compare-nest — currently proven only
  indirectly by gated load/store goldens).
- WMMA estimate unit test (flops factor covered only by goldens).
- Re-verify `bf16_vector_load_reindexes_shrink` (decomp) end-to-end: it
  relies on the LOAD inheriting the shrink's width-2 shape.

## Misc

- `Uop.program_vals` raises bare `Not_found` for an unbound variable; the
  reference raises a descriptive error naming the variable and its user.
