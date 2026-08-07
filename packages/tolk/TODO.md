# TODO

Known gaps, deferred parity items, and follow-ups from the 2026-07 parity
campaign (reference: `_tinygrad` @ 7eb197b1b). Maintainer notes — reference
anchors point at the tinygrad clone.

## Open bugs

- **Whole-vocab-axis vectorized store on CUDA**: the float16 GPT-2
  loss-multiply backward renders `make_float50257(...)`, which NVRTC rejects
  (recorded in Kaun's GPT-2 `train.ml`). Not reproduced: the same model shape
  and backward form compile clean on CPU at `vocab = 50257`, and the case needs
  a CUDA device.

  The optimiser is ruled out as the source. A 50257-wide lane count would have
  to come from an UPCAST or UNROLL of the vocab axis, and neither can reach it:
  `apply_opt` resolves an amount of `0` to the axis's full extent, but then
  `validate_shift_opt` caps the *resolved* amount at 16 for UPCAST and 32 for
  UNROLL, so both are rejected. No sequence of beam actions produces it.

  That leaves the shape itself: a node of shape `[50257]` reaching devectorize
  without a range behind it, which `do_devectorize` turns into one STACK lane
  per element (an EXPAND on a scalar becomes `U.stack (List.init n ...)`). The
  broadcast of the scalar loss back over the vocab axis is the candidate — the
  search should start at whatever leaves that axis rangeless in the backward
  kernel, not at the renderer, which sizes a STACK correctly by construction
  (`stack_count` is `Array.length srcs`).

## Deferred parity divergences (each with a known blocker)

- **Index factoring in tensor-core kernels**: the 32³ matmul under
  `TC:0:-1:0:1 + UNROLL:0:0` is byte-identical to the reference except that the
  reference folds one thread-index expression the symbolic pass leaves split —
  it emits `((lidx0>>4)<<7)+(((lidx0>>1)&3)<<5)` where tolk emits the two bit
  extracts `(((lidx0>>2)&1)<<6)` and `(((lidx0>>1)&1)<<5)` separately. Owner is
  the symbolic pass, not the renderer. This blocks a tinygrad-backed
  exact-source golden for tensor cores, so the WMMA width invariant is pinned
  by renderer unit tests instead (`test/unit/test_cstyle.ml`).
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

- **Reconstructed sizes with silent defaults**: in several places tolk
  recomputes a width or extent that the reference reads straight off the node,
  and each reconstruction carries a fallback for the case it does not model.
  Every gap found so far has produced a silent miscompile rather than a visible
  diff, because the fallback yields a plausible number instead of failing.
  Three instances:
  - `expr_numel` (`renderer/cstyle.ml`) consults the node's shape only for
    `Index`/`Shrink`/`Wmma` and otherwise propagates from sources, defaulting
    to one lane. The reference just uses `u.max_numel()`. `Wmma` was missing,
    so every WMMA declared scalar and tensor cores were uncompilable on all
    backends.
  - `range_int_size` (`schedule/rangeify.ml`) returned `1` for anything that
    is not a `RANGE`, so a stage flattened to a single index expression sized
    to one element. Fixed.
  - `stage_to_store` (`schedule/rangeify.ml`) sizes a bufferize from two
    sources and silently takes the second when they disagree; only a
    non-positive product is rejected. Still present, and it swallowed the
    late-allreduce overflow rather than reporting it.

  The question this raises is whether these collapse into reading the node's
  own shape, and if some op genuinely needs source propagation, which and why.
  Each is extra machinery relative to the reference, so it has to earn its
  place. Renderer- and scheduler-wide, so it wants its own pass rather than a
  fourth point fix.
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
- Validity-clause ordering in `simplify_valid` is uncovered. Inverting the
  `_valid_priority` sort comparator (`lib/uop/symbolic.ml:2259`) leaves the uop
  unit suite, every golden, and parity green, so nothing in the corpus
  discriminates on clause order. Left open deliberately: the priority is a
  simplification-order heuristic, so a wrong order costs simplification rather
  than correctness, and a synthetic case would pin today's ordering instead of
  the property.

## Misc

- `Uop.program_vals` raises bare `Not_found` for an unbound variable; the
  reference raises a descriptive error naming the variable and its user.
