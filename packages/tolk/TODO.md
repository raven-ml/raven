# TODO

Known gaps, deferred parity items, and follow-ups from the 2026-07 parity
campaign (reference: `_tinygrad` @ 7eb197b1b). Maintainer notes — reference
anchors point at the tinygrad clone.

## Open bugs

- **A scalar bufferize keeps a ranged shape and stores every lane to one
  offset**: an `INDEX` with *no* index sources onto a `STAGE` that carries a
  loop range keeps the range in its shape, because `Uop.shape` for `Ops.Index`
  is the index shapes followed by the un-indexed tail of the pointer's shape.
  `broadcast_binary` then widens the store value to that many lanes and
  `do_devectorize` splits it into one store per lane at a single offset, which
  the coalescer rejects (`Coalese: multiple stores to the same offset`). With
  `DMC=1` the whole program compiles and no STACK anywhere exceeds 8.

  This is the live defect behind the `make_float50257(...)` NVRTC rejection
  recorded in Kaun's GPT-2 `train.ml`. That symptom is stale: the wide store
  now dies in the coalescer, earlier than the renderer. At vocab 50257 the
  same arithmetic produces the 50257 in the recorded name.

  The trigger is the *forward* loss unscale, `Nx.div loss scale`, not the
  loss-multiply backward: the scalar `1/scale` is shared between a scalar
  consumer and the vocab-shaped gradient unscale, and it only fires when the
  scalar consumer is a reduce output fused into the same kernel. Switching
  loss forms fixed it by removing the division, not the multiply. Reproduces
  in about a minute at `VOCAB=1009 EMBD=64 LAYER=1 HEAD=4 INNER=256 BS=2
  SEQ=8`, and a ~20-line hand-built kernel AST reproduces it at any width.

  It needs **neither a CUDA device nor CUDA**: `lib/schedule/**` and
  `callify.ml` reference no renderer, so kernel boundaries are
  renderer-independent, and the CUDA renderer drives
  `full_rewrite_to_sink`/`Renderer.render` with no GPU present. It fails on
  CPU too, at any vocab width including 64.

  Owner is `schedule/rangeify.ml:829` (`stage_index_sources`), which returns
  `None` on the STAGE/INDEX arity mismatch and keeps the stage. The reference
  asserts the case cannot arise — `rangeify.py:217`, `assert len(buf.src) ==
  len(idx.src), "index on wrong bufferize"`, under the comment `# see if we
  can't do it, should this ever hit?`. Restoring the assert only relocates the
  failure, and tolerating the short index papers over it: the real question is
  why a rank-0 node carries two out-ranges. `indexing.ml:584`
  (`wrap_realized_src`) is line-for-line the reference and is *not* the
  divergence; `choose_consumer_rngs`/`merge_consumer_rngs`
  (`indexing.ml:355-430`) differ structurally — tolk has a rank-mismatch
  branch that realizes on all own axes, and its `transpose` indexes to the
  length of the first consumer list where Python's `zip` truncates to the
  shortest. Which of those produces the rank-2 entry is not yet pinned.

- **`Coalese: multiple stores to the same offset` under the CUDA renderer
  only**: a cross-render probe found kernels that render cleanly on CPU and
  fail in the coalescer when the same kernel ASTs are lowered through the CUDA
  renderer. Distinct from the entry above, which fails on both. Not yet
  minimised.

## Deferred parity divergences (each with a known blocker)

- **`pm_index_invalid` is kept, though the reference deleted it** (`557e67486`).
  `lib/uop/symbolic.ml`. The rule drops the validity gate on a cast or a
  comparison over an index-domain gated index, on the grounds that the consumer
  recovers the gate itself with `Uop.get_valid`. The reference can delete it
  because `data_srcs` and the rangeify rework keep those predicates apart
  structurally; tolk has neither yet, so deleting it does not converge on the
  reference — it ships broken RNG.

  **Removal condition**: delete `pm_index_invalid` when `data_srcs` lands
  (wave 6-1 of the pin update), and re-run `test/unit/frontend/test_rand.ml`.
  That suite is the tripwire: it is the only place in the tree where the
  failure is reachable, and it fails loudly (every draw collapses to a
  constant), so a green run after removal is sufficient evidence that the
  structural mechanism has replaced the dtype one.

  **Evidence.** Without the rule, `rand` returns exactly 0, `randint` returns
  all `low`, and `randperm` returns the identity — 14 of 25 rand tests, fully
  deterministic. The rand kernel is emitted *empty*: no output buffer, no body.
  Store-count instrumentation puts the loss in `codegen.ml`'s `initial
  symbolic`, and the sink entering that stage carries the store gate

      AND(CMPLT(range, 2), CMPNE(CMPLT(range, 2), true))

  i.e. `v AND NOT v` on the *same* node — identically false. `sym` then folds
  the gate to false and `pm_invalid_load_store` correctly deletes the store, so
  every stage downstream of the gate is behaving correctly; the defect is that
  an unsatisfiable gate was built at all. Mechanism: `rand` builds its bits
  with `Op.cat`, so one half's index is gated by `r<2` and the other by
  `NOT(r<2)`. With the rule gone, a comparison over a gated index lifts its
  gate into the surrounding boolean algebra via `pm_data_invalid`, the two
  halves' predicates meet in one conjunction, and they annihilate.

  **Re-keying.** The pre-migration rule keyed off the `Invalid` sentinel's own
  dtype (`sentinel.dtype = index`). `Invalid` is now a bool const with no dtype
  of its own, so the rule keys off the `where` node's dtype being `weakint`
  instead. These agree wherever the gate was built by `Uop.valid`, which
  propagated the gated value's dtype onto the sentinel. Verified rather than
  argued: instrumenting the predicate across the whole unit suite reports
  exactly one site where the pattern matches and the where-node is not
  `weakint`, and it is `test_symbolic.ml`'s
  `invalid_gate_comparison_gates_nonweak_invalid`, which deliberately builds an
  `int32` sentinel and asserts the gate is *preserved* — both keyings agree it
  must not fire there. The two keyings can still differ in principle for a gate
  built as `where(c, x, Uop.invalid ())` with a non-index `x`, since the old
  sentinel defaulted to `index` regardless of `x`; no such site is exercised by
  the suite, and the new behaviour there (keep the gate) is the conservative
  one.

  **Hypotheses eliminated while diagnosing this** — each toggled in isolation
  and re-run against `test_rand.ml`; none is the cause, so do not re-tread
  them: the `Int64 -> Weakfloat` promotion-lattice edge; `Const.float`
  truncating on construction; dropping the `.cast(dtype)` wrappers on `Invalid`
  in `pm_data_invalid`; adding `Ops.Cast` to those rules; widening the
  gate-lifting rules from non-comparison binaries to all binaries; and skipping
  `Invalid` in `skip_for_rangeify`.

- **`skip_for_rangeify` also skips the `Invalid` sentinel**
  (`lib/schedule/indexing.ml`). Correct independently of the rand bug above,
  and not a workaround for it: `Const.invalid` used to default to an *index*
  dtype, so rangeify skipped it as an index-domain node. `Invalid` is now bool,
  so without an explicit test it would start taking part in range propagation
  — and propagating ranges through a gate's `Invalid` branch conjoins the gate
  with its own negation. Subsumed by `data_srcs` in wave 6-1, which excludes
  such positions structurally.

- **The host-scalar `bitcast` stays in `symbolic.ml`** rather than moving to
  `dtype.ml` as the reference does (`67dc02d7e`). Forced, not skipped: tolk's
  equivalent (`bitcast_const_storage`) takes a `Const.t`, and `Const` depends on
  `Dtype`, so the move is impossible without re-cutting it to `storage_scalar`
  — for a consumer tolk does not have, since `lib/frontend/rand.ml` builds
  `_bits_to_rand` out of UOp `Bitcast` nodes rather than host-side arithmetic.

- **Index factoring in tensor-core kernels**: the 32³ matmul under
  `TC:0:-1:0:1 + UNROLL:0:0` is byte-identical to the reference except that the
  reference folds one thread-index expression the symbolic pass leaves split —
  it emits `((lidx0>>4)<<7)+(((lidx0>>1)&3)<<5)` where tolk emits the two bit
  extracts `(((lidx0>>2)&1)<<6)` and `(((lidx0>>1)&1)<<5)` separately. Owner is
  the symbolic pass, not the renderer. This blocks a tinygrad-backed
  exact-source golden for tensor cores, so the WMMA width invariant is pinned
  by renderer unit tests instead (`test/unit/test_cstyle.ml`).

  Not tensor-core-benchmark-specific: it reproduces unchanged on every Metal
  tensor-core kernel of the `rnn_grad` reverse pass and on the 128³
  `matmul_small` and `attention` bench workloads, in the same shape
  (`(((lidx0>>2)&1)<<8)+(((lidx0>>1)&1)<<7)` against the reference's
  `((lidx0>>1)&3)<<7`). In each of those the kernel name and launch dims are
  byte-identical, so the axis structure agrees and only the fold is missing.
  Verified identical under both `7eb197b1b` and `baa614806`, so it is not pin
  churn.
- **Accumulator numbering**: tolk numbers reduce accumulators in a different
  order than the reference, so kernels with more than one reduce render the
  same code with `buf`/`acc` indices permuted. Semantically inert — the
  declaration set is identical and the bodies match under a consistent
  bijection — but it is a real source divergence and the last residue of the
  aliasing miscompile fixed above: the pre-pass that used to hide it was
  deleted, since it was preventing a cosmetic diff at the cost of a silent
  miscompile.

  Four cases are red on it: `test/parity/group_reduce_pair` (swap of the two
  scalar accumulators, shared staging buffer unchanged),
  `test/parity/wide_reduce_thread` (exact reversal of eleven accumulators,
  reference `10, 9, … 0` against tolk `0, 1, … 10`), `test/parity/rnn_unroll`
  (all four backends) and `test/parity/two_sum` (visible at both stage 5, as
  the `ParamArg(N, …, addrspace=REG)` slot, and stage 7). **Do not regenerate
  these** — the numbering is the finding.

  The exact reversal in the eleven-accumulator case is the strongest hint: the
  fix is to match the reference's `pm_reduce_local` visitation order, not to
  re-derive the numbering.
- **WMMA accumulator element order**: the register array holding a tensor-core
  accumulator is laid out in a different axis order than the reference. On the
  128³ Metal matmul the reference orders it, outermost to innermost, as
  [second operand (4), WMMA output lane (2), first operand (4)] — the lane
  axis at stride 4 — while tolk emits [first operand (4), second operand (4),
  lane (2)], the lane innermost at stride 1. **Not a miscompile**: every
  stored address receives the same (operand pair, lane) on both sides, so the
  two layouts are a pure permutation of the local array.

  The layout follows the axis positions in `build_range_map`
  (`codegen/codegen_lower.ml:137`), which numbers UPCAST/UNROLL ranges in
  toposort order. For that kernel the reference's map is `{8→0, 3→1, 7→2}`
  with sizes 4, 2 (the lane) and 4; tolk's rendered strides correspond to
  `{8→0, 7→1, 3→2}` — the same three ranges with the lane moved last. The WMMA
  `upcast_axes` construction in `opt/postrange.ml` (including
  `with_missing_tc_axes`), `contract_axis`/`unroll_axis` and `Uop.toposort`
  all mirror the reference, so the divergence is in the order those ranges
  reach `build_range_map`. Owner is `opt/` or the expander, not the renderer.

  Invisible on the 32³ tensor-core case above, which has a single WMMA and so
  cannot observe the ordering — that case being otherwise byte-identical is
  not evidence of layout parity. Verified identical under both `7eb197b1b` and
  `baa614806`.
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

- **Reconstructed values with silent defaults**: in several places tolk
  recomputes something the reference reads straight off the node or produces
  as it goes, and each reconstruction carries a fallback for the case it does
  not model. Every gap found so far has produced a silent miscompile rather
  than a visible diff, because the fallback yields a plausible value instead
  of failing. Four instances:
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
  - `reduce_slots_in_tinygrad_order` (`codegen/codegen_lower.ml`) precomputed
    an accumulator slot per REDUCE where the reference simply bumps a counter
    as the rewrite reaches each one; REDUCEs created mid-rewrite fell through
    to a shared counter and collided, so two accumulators aliased and a kernel
    silently doubled its result. An ordering reconstruction rather than a size
    one, but the same shape of bug. Deleted in favour of the reference's
    counter; see the accumulator-numbering entry above for what that left
    behind.

  The question this raises is whether these collapse into reading the node's
  own shape, or into doing what the reference does as it goes. Each is extra
  machinery relative to the reference, so it has to earn its place. Renderer-,
  scheduler- and codegen-wide, so it wants its own pass rather than a fifth
  point fix.
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
