# TODO

Known gaps, deferred parity items, and follow-ups from the 2026-07 parity
campaign (reference: `_tinygrad` @ 7eb197b1b). Maintainer notes — reference
anchors point at the tinygrad clone.

## Open bugs

- **`Coalesce: multiple stores to the same offset` under the CUDA renderer
  only**: a cross-render probe found kernels that render cleanly on CPU and
  fail in the coalescer when the same kernel ASTs are lowered through the CUDA
  renderer. Not yet minimised.

  The scalar-bufferize entry that used to sit above this one is **fixed**: the
  apply-rangeify pass was copying the realize and range maps onto the nodes it
  rebuilt, and hash-consing collapses nodes that differed only in the movement
  ops rangeify removes — so a rank-2 value's ranges reached a rank-0 rebuild
  and its consumer indexed a STAGE with too few indices. The maps are now
  keyed on the walked graph only, as in the reference. Reproducer for the
  future: the gpt2 float16 scaled step at `VOCAB=1009 EMBD=64 LAYER=1 HEAD=4
  INNER=256 BS=2 SEQ=8`, which used to die in the coalescer and now trains.

## Deferred parity divergences (each with a known blocker)

- **COPY still takes part in rangeify** (`f65001e29`, `dcad11941`). Upstream
  converts every COPY to a BUFFER+STORE before rangeify (`pm_copy_to_store`),
  drops COPY from `ALWAYS_CONTIGUOUS`, from the realize map and from
  `ALWAYS_RUN_OPS`, lets `split_store` emit a plain kernel for it, and rebuilds
  the COPY afterwards in `schedule/__init__.py` (`pm_copy_from_store`,
  `copy_kernel_to_copy_uop`, `simplify_copy_kernel`) with
  `assert_all_same_devices` replacing the device check inside `split_store`.

  **Blocker**: the second half lands in `lib/engine/realize.ml` (copy
  detection) and needs `pm_simplify_ranges`/`pm_flatten_range` exported from
  `lib/codegen/simplify.ml` — neither is in this wave's ownership, and both
  were being rewritten concurrently. It also moves every rangeify golden, so
  it wants to land in one commit with a golden regeneration.

- **Kernel formals keep their caller-side variable names** (`be25207a7`).
  Upstream renames a BIND'd scalar formal to `p{slot}` in `UOp.param_like` and
  maps it back in `resolve_linear_call` with a `substitute(..., enter_calls)`.
  tolk identifies kernel-body variables by name and address space
  (`engine/schedule.ml`, `variables_of_kernel_body`), so two callers that bind
  different values to the same variable name are not distinguished.

  **Blocker**: `param_like` lives in `lib/uop/uop.ml`. No tolk failure is known
  to depend on it — the divergence is scoping hygiene, not a live bug.

- **Custom-kernel inputs are not force-realized** (`470c032a5`, `f253c4469`).
  Upstream's `realize_custom_kernel_srcs` realizes the arguments of a CALL
  whose body is a SINK or PROGRAM and marks them non-removable. tolk has no
  entry point that builds such a CALL in a tensor graph (no `custom_kernel`),
  so the rule and the `non_removable` set it feeds have no consumer. Port both
  with the feature, not before.

- **`Symbolic.pm_fold_cast_const` is composed at one of its six sites.**
  The rule that collapses `CAST(dt, CONST v)` into a typed constant was split
  out of `symbolic_simple` (`d726e5f7f`, pruned by `d51e55aa1`) because it
  writes a strongly typed constant that weak-dtype lowering must be free to
  decide for itself. The reference re-composes it at exactly six places, and
  deliberately not at the others; until the rest land, any pass that used to
  get the fold for free from `sym`/`symbolic_simple` no longer does, and its
  rendered output carries casts the reference folds away.

  `schedule/rangeify.ml`'s `post_rangeify_rules` has it. The five remaining:
  `codegen.ml` "initial symbolic", "devectorize2", "early symbolic", and
  `pm_decomp`; and `codegen/simplify.ml`'s substituted range rewrite. It must
  *not* be added to "postopt symbolic", "expand broadcast / add loads", "add
  images", "extra symbolic", "final symbolic", or `pm_reduce_collapse`.

- **The reference's UNSHARD, COPY, CALL, void-RANGE, and WMMA spec rules are
  not in `lib/uop/spec.ml` yet.** Each waits on the feature it describes:
  `Ops.Unshard` is named but still carries the old single-axis `Multi` arg
  shape, so its spec rule waits on the sorted-axis-tuple rework; COPY
  keeps `allow_any_len` until copies leave rangeify (`dcad11941`); the
  address-valued `CALL` rule needs call-inside-C (`6e979b879`); the void-RANGE
  loop header and the `END(body, range, cond)` backedge need the loop rework
  (`b764599d8`, `39924387b`); and the 5-tuple WMMA arg check needs the WMMA arg
  shrink. Port each spec rule with its owning change, not before — a spec that
  admits a shape nothing builds proves nothing.

- **The threefry symbolic rules are not pruned yet** (`33755a346`).
  `lib/uop/symbolic.ml` still carries five unpack rules (the `&0xFFFFFFFF`
  cast, both `*2^32` splice forms, and both `<<32` forms); the reference keeps
  only two. This was one third of a three-file change; two thirds have landed —
  `threefry2x32` in `lib/codegen/decomp/decomp_op.ml` now splits, rotates and
  repacks with `<<`/`>>` and narrows the low word by cast rather than mask, and
  `_threefry_random_bits` in `lib/frontend/rand.ml` no longer masks before its
  narrowing casts. All that remains is pruning the three now-dead symbolic
  rules.

- **`Uop.wait` and its spec rule are not deleted yet** (`f41e4a758`). Two
  recorded claims about this commit are wrong. The recon said "zero hits in
  tolk": in fact `Uop.wait` constructs one, `Uop.as_wait` views one, a
  `spec.ml` rule admits one, and `test/unit/uop/test_uop.ml` names one. And it
  is *not* a whole-op deletion — the reference leaves `WAIT` in the `Ops` enum
  and removes only `UOp.wait`, the WAIT arm of `dtype_from_uop`, and the
  `spec.py` rule. So `Ops.Wait` stays (see the note under Misc); what remains
  to delete spans `uop.ml{,i}`, `spec.ml`, and the two test suites.

- **`has_index` duplicates `op_in_backward_slice_with_self` because
  `backward_slice` is uncached.** Both feed `fold_where_closure` in
  `lib/uop/symbolic.ml`. The reference expresses that guard as
  `u.op_in_backward_slice_with_self(Ops.INDEX)`, which costs nothing there
  because `backward_slice` is a cached node property; tolk's
  (`lib/uop/uop.ml`) is a plain `toposort` filter and `in_backward_slice`
  re-toposorts per call, so routing a rule that runs on every `where` in every
  pass through it would be quadratic. Hence the local memo. Both collapse into
  the node API once `backward_slice` is cached — which is a change to a hot
  path and has to be measured against the current rewrite engine, so it is its
  own piece of work, not a rider on whichever wave notices it.

  (`bool_slice` itself has moved to `lib/uop/uop.ml`, where the reference keeps
  it, exposed as a membership query rather than a table.)

- **`PARAM // c` is not treated as irreducible** (`980748ccf`).
  `lib/uop/divandmod.ml`. The reference short-circuits `fold_divmod_general`
  when the numerator is a `PARAM` whose `multiple_of` is divisible by the
  denominator. Tolk's `param_arg` has no `multiple_of` field. It is not
  renderer work — the field is never rendered — so it lands with the field
  itself; see item 3 of the post-wave sweep, which also records that no
  existing test would catch a botched threading.

- **The host-scalar `bitcast` stays in `symbolic.ml`** rather than moving to
  `dtype.ml` as the reference does (`67dc02d7e`). Forced, not skipped: tolk's
  equivalent (`bitcast_const_storage`) takes a `Const.t`, and `Const` depends on
  `Dtype`, so the move is impossible without re-cutting it to `storage_scalar`
  — for a consumer tolk does not have, since `lib/frontend/rand.ml` builds
  `_bits_to_rand` out of UOp `Bitcast` nodes rather than host-side arithmetic.

- **A tensor-core source golden is now possible and does not exist.** The index
  factoring that used to block it is fixed (`fold_add_divmod_recombine` now
  re-bases a quotient partner through a merged divisor, so two adjacent
  single-bit extracts of a thread id collapse into the one multi-bit extract
  the reference emits). Nothing under `test/golden/codegen` or `test/parity`
  covers a tensor-core kernel's source, so the WMMA width invariant is still
  pinned only by renderer unit tests (`test/unit/test_cstyle.ml`). Generate one
  from the clone: the 32³ matmul under `TC:0:-1:0:1 + UNROLL:0:0` is the case
  the divergence was first found on, and the Metal `rnn_grad` reverse pass,
  `matmul_small` 128³, and `attention` all exercise the same shape.

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
- **expand_bitcast** (rangeify): a BITCAST between dtypes of different
  itemsize has to be re-expressed as a reshape, a per-part shift and a
  recombination. It blocks half-precision `Rand.rand`; no golden exercises it.

  Still missing, both in `lib/uop/`, not in `lib/schedule/`: an axis-adding
  `stack ~dim` (tolk's `Uop.stack` builds a vector/shape tuple, not a new
  axis) and `flatten` over a negative axis range. `squeeze` is expressible as
  a `Uop.reshape`, and `Uop.shrink`/`cast`/`usum`/shift cover the rest — so
  the rule is roughly ten lines once those two land. Reference:
  `rangeify.py:116` plus `mixin/movement.py`.
- **Host I/O via allocator bridge** (`frontend/run.ml`): upload/download use
  `Buffer.copyin`/`as_bytes` directly because tolk has no host pseudo-device
  to route a `copy_from` through (reference: device.py:113-115 seeds via a
  PYTHON-device buffer). Converges to `copy_from` if a host device lands.
- **Image coordinate form**: the gater/coalesce/renderer accept both
  two-axis `INDEX(buf, y, x)` (reference form) and stacked
  `INDEX(buf, STACK[y,x])`. Converge on two-axis and drop the stacked branch,
  gated on an image golden generated from the clone (none exists — the image
  path is uncovered by committed goldens). Tolk's own `transform_to_image`
  emits the stacked form, so that branch is load-bearing today: converging
  means changing the producer, `coalesce.ml`'s `indexing_simplify`,
  `gater.ml`'s `indexed_two_invalid_gate`, and the renderer in one commit.

  *What catches a bad convergence*: `test/unit/codegen/test_images.ml` covers
  the pieces — image dimension selection, `pm_simplify_add_image` turning a
  SHRINK into an image index, and the coalescer — but every case is
  hand-built, so it pins tolk's own form rather than the reference's. **No
  committed golden renders an image kernel**, which is why the convergence is
  gated on generating one first: without it the suite would keep passing on
  whichever form the change happened to produce.

- **`Uop.O` arithmetic does not promote, so the reference's explicit casts
  stay** (`c9e11544d`). The reference deleted `.cast(...)` from a dozen
  codegen expressions because `UOp.__add__` and friends route through
  `_broadcasted`, which casts both operands to their least upper dtype.
  Tolk's `Uop.O` operators are thin `alu_binary` wrappers: the node dtype is
  the *left* operand's, with no promotion. Dropping a cast whose weak or bool
  operand sits on the left therefore mistypes the node. The casts stay in
  `codegen/simplify.ml` (`fold_result`, `rule_lift_add_lt`,
  `rule_reduce_and_where`), `decomp/decomp_dtype.ml` (`l2i` ADD/SUB carry,
  `rne`), `decomp/decomp_op.ml` (`floordiv_to_idiv`), and
  `decomp/decomp_transcendental.ml` (`xexp2`). The one arm already ported is
  `reduce_unparented`'s range-size multiplier, where the strong operand is on
  the left so both rules agree. Unblocks with `dtype_from_uop`/`promo_dtype`
  on the node type.

  *What catches a bad port*: **nothing that currently runs.** `lib/uop/spec.ml`
  has the rule that would reject it — `ops Ops.Group.alu =?> Array.for_all
  (matches_or_weak u)`, which admits a source only if it matches the node's
  dtype, is `Invalid`, or is weak — but `Spec.type_verify` is gated on
  `SPEC=1` and **no suite sets `SPEC`**. A mistyped ALU node would reach the
  renderer and emit plausible C. Whoever lands `promo_dtype` should turn
  `SPEC=1` on for at least one suite in the same commit, or this deletion has
  no guard at all.

- **The WMMA arg is still an 8-tuple** (T6: `61e104bdf`, `2b1146b3f`,
  `d8b83daac`). The reference shrank it to
  `(dims, dtype_in, device, threads, tc_upcast_axes)` and derives the name and
  output dtype; `expand_wmma` then gates on `arg[4] is not None` and clears
  that slot instead of clearing the node tag. Tolk's `expand_wmma`
  (`codegen/codegen_lower.ml`) still gates on `node_tag = Some "1"`. The
  change starts at `Arg.Wmma_info` in `lib/uop/uop.ml{,i}` and sweeps
  `renderer/cstyle.ml`, `device.ml`, `program_spec.ml`, `uop/spec.ml`, and
  `opt/postrange.ml`; it cannot land from `codegen/` alone.

  *What catches a bad port*: `test/unit/codegen/test_tc.ml` pins every table's
  WMMA names (which the reference now derives from the arg rather than storing);
  `test/unit/test_cstyle.ml` pins the WMMA accumulator width; and
  `test/parity/tc_matmul_{f16,bf16,fp8}` render tensor-core kernels end to end
  on four backends. Between them the derived name, the declared width and the
  emitted call are all covered.

- **`pm_device_to_var` is not in `codegen/gpudims.ml`** (`7d4892629`). The
  reference lowers an `AxisType.DEVICE` range to the `_device_num` variable at
  the end of `pm_add_gpudims`, and drops that range from the `END`s that
  closed it. Tolk has neither `AxisType.Device` nor `Ops.Unshard`, so there is
  no DEVICE range to lower. Lands with the MULTI → UNSHARD rework, together
  with the matching DEVICE guards in `codegen/simplify.ml`'s `mark_range_mod`
  and `opt/postrange.ml`'s `rngs`.

  *What catches a bad port*: `test/unit/engine/test_multi.ml` and the twelve
  `test/parity/multi_*` cases, which dump both the scheduled graph (stage 5)
  and the rendered source (stage 7) — a DEVICE range that reaches the opt
  axes or fails to become `_device_num` shows up in both.

- **The void-RANGE loop header is not ported** (`b764599d8`, `39924387b`).
  Beyond the spec rules noted above, the reference threads a "backedge" (the
  void range plus its bool condition) through `flatten_range` and
  `do_split_ends`, guards `simplify_merge_adjacent` against a non-RANGE ended
  child, and filters void ranges out of `Scheduler.rngs`. The feature exists
  for HCQ wait loops; nothing in tolk constructs a void range, so all four
  branches would be unreachable. Port them with a producer, not before.

  *What catches a bad port*: **nothing, and nothing can** — there is no way to
  build a void range, so no test can reach the branches. The producer and the
  branches have to land together, with the producer's own test.

- **`multi_pm` does not run at the head of `full_rewrite_to_sink`.** The
  reference resolves in-kernel shards (register fragments) during codegen, not
  only in the scheduler. Tolk's `multi_pm` needs scheduler-supplied shape and
  device callbacks and there is no in-kernel `Ops.Multi` to resolve, so the
  pass would be a no-op traversal. Lands with UNSHARD and `alloc_fragment`.

  *What catches a bad port*: **nothing today** — with no in-kernel shard to
  resolve, adding the pass is unobservable either way. It becomes testable
  only once `alloc_fragment` can put a shard inside a kernel; that commit
  needs the test.

- **`52c9e5a99` must land in two commits, in this order. This is required
  procedure, not advice.** The reference renames the old `LOOP` to `WEAK`
  *and* adds a new `LOOP` for the void-RANGE header. Doing both at once
  type-checks perfectly while silently changing the meaning of every existing
  `Loop` site — the one refactor no compiler error can catch.

  1. **Rename only.** `Axis_type.Loop` → `Axis_type.Weak` tree-wide, adding
     no constructor. Every reference either updates or fails to build, so
     there is no window in which a stale reference means something new.
  2. **Then add the new `Loop`,** as a separate commit. By then nothing
     refers to `Loop`, so it starts with no inherited meaning.

  **Step 1 has landed** (`lib/uop/axis_type.ml` reads
  `Global, Warp, Local, Weak, Group_reduce, …`; `lib/codegen/**`,
  `lib/schedule/**` and the unit suites were swept with it). Step 2 is still
  open and must stay a separate commit.

  *What catches a bad sweep*: `test/unit/codegen/test_postrange.ml` builds
  `Weak` ranges (`weak_rng`) and drives UPCAST, LOCAL and THREAD through
  them, asserting the resulting axis kinds. If `upcastable_dims`, the
  UPCAST/LOCAL preconditions, or `_globalizable_rngs` read the new `Loop`
  where they mean `Weak`, those opts start being rejected and that suite goes
  red. `codegen_lower.ml`'s `range_repeats` is additionally written as an
  exhaustive match over `Axis_type.t`, so adding a constructor in step 2
  fails to compile there and forces an explicit decision.

- **Frontend promotion still broadcasts shapes and has no weak scalar
  literal.** The reference's `_broadcasted` (`40f0d4af1`, `9433790ad`,
  `b1a72299a`) is dtype-only: it promotes to `least_upper_dtype`, keeps a weak
  CONST weak at `weak_dtype(out)`, and leaves shapes alone for
  `pm_expand_broadcast` to widen in codegen. tolk still expands shapes in
  `Op.broadcasted` behind the `Tensor.broadcasted_hook` indirection, and
  `Tensor.i`/`f`/`b` build strongly typed default-dtype constants. Two
  blockers: `alu_binary` derives its dtype from the left source rather than
  `promo_dtype(src)`, so two differently typed operands would silently take the
  left one; and no `pm_expand_broadcast` exists yet in `codegen_lower.ml`.
  Landing it deletes `Tensor.broadcasted_hook`, moves `broadcasted` into
  `elementwise.ml` where the reference keeps it, and dissolves the triplicated
  `uf` helper below. Carries three riders that must land in the same sweep:
  `b1060ca70` (`_pad_constant` stops promoting `base`, because the weak fill
  value promotes it instead — removing the promotion first would silently
  truncate a float fill into an int tensor), `46b82d475`/`f19a2ad77` (the
  single `where` mixin), and `f5d9c31d1`'s remaining half.

- **`cat` does not take the same-shape `stack` fast path** (`250de4b14`,
  final state `e684fcc68`). The reference's `stack` is a movement op
  (`_mop(Ops.STACK)`) and `cat` delegates to it when every input has the same
  extent on the cat axis. tolk inverts the relationship: `Op.stack` is
  `unsqueeze` + `Op.cat`, so the reference's fast path would recurse. Needs
  `Ops.Stack` as a tensor-level movement op first; `Uop.shape` already gives it
  the right shape (`(len src) :: shape src.(0)`), but nothing constructs one
  from the frontend. The `pm_reduce_collapse` half of `e684fcc68` is wave 5-2's
  and independent.

- **`Renderer` has no `abi` field and no CALL-in-C rule** (`d1f215d37`,
  `6e979b879`). Both exist upstream for the CPU uop worker, a runtime tolk does
  not port. `abi` is `"__attribute__((ms_abi)) "` on win32 and empty elsewhere;
  with CALL-in-C absent its only use is `kernel_typedef = abi + "void"`, which
  tolk already spells statically. CALL-in-C renders a function-pointer cast for
  an address-valued `CALL`, which needs `ret_dtype` on `Uop.call`, a
  `dtype_from_uop` arm, a `_shape` arm, and a spec arm — all to render a node
  no tolk pass constructs. Port them together if a host-callable runtime lands.

- **`ParamArg` has no `volatile` flag** (`a6fda6b10`). Its only producers
  upstream are `ops_cpu.py`, `ops_qcom.py`, and `hcq2.py`; its only consumers
  are the cstyle parameter list and the LLVM renderer. tolk would carry an
  always-false field. Add it with the first runtime that needs uncached
  parameter memory.

- **`ProgramInfo` carries no `target`, and `aux` is deleted rather than
  renamed** (`9fdaa4bff`). The reference swaps `aux` for `target` so
  `UOp.to_elf()` can build a `TinyELF`. That is the runtime restructure tolk
  does not port — tolk dispatches through `lib/compiler.ml`, and
  `Program_spec.t` already carries `device` — so `target` would be a field
  with no reader, which is porting bloat rather than parity. tolk's `aux` had
  no reader either: its only producer was `OpenCLRenderer.aux`, and nothing in
  `lib/runtime/` consumed it; it reached only the diskcache key and the debug
  repr. So the whole channel is gone: `Renderer.has_aux`/`aux`, the
  `?aux` parameters on `Renderer.make` and `Program_spec.of_program`,
  `Cstyle.opencl_aux`, the `device.ml` cache-key line, the `codegen.ml`
  producer, and the `render.ml` repr entry. **`Uop.program_info.aux` and the
  `?aux` on `Uop.program_info_from_sink` are the last two sites** — see the
  post-wave sweep below.

- **`_broadcast_to` computes its expand axes by hand.** The reference uses
  `broadcast_axes(shape, new_shape)` (`f64f96ec5`), which tolk's uop layer does
  not export. Equivalent for concrete shapes, which is all `Movement` accepts.

- **Negative slice bounds against a symbolic size** (`mixin/movement.py`). The
  reference resolves `start`/`stop` against a possibly symbolic `size` before
  deciding whether the slice is fully concrete. `Movement.parsed` is built from
  `Tensor.shape`, which raises on a symbolic dimension, so tolk cannot reach
  the case at all. Unblocks with symbolic-shape `getitem`.

## Post-wave sweep

Four items that no single wave can land, because each one changes a
declaration in `lib/uop/` and every reader of it across `lib/codegen/`,
`lib/renderer/`, `lib/schedule/` and the suites. Each needs one owner touching
all of those files in one commit, after the waves settle. For each: what test
goes red if it is done wrong. Where the honest answer is "none", the test comes
*before* the sweep, not after.

- **1. `Axis_type`: add the new `Loop`.** Owner: to be assigned. Step 1 (the
  `Loop` → `Weak` rename) has landed; step 2 adds the void-RANGE header kind
  and must stay a separate commit, for the reason spelled out under
  `52c9e5a99` above.

  *What catches it*: `test/unit/codegen/test_postrange.ml` drives UPCAST,
  LOCAL and THREAD through `Weak` ranges and asserts the resulting kinds, so a
  site that reads the new `Loop` where it means `Weak` starts rejecting opts
  and goes red. `codegen_lower.ml`'s `range_repeats` is an exhaustive match, so
  it additionally fails to compile and forces an explicit decision per site.

- **2. `wmma_info`: 8 fields → the reference's 5-tuple** (T6). Owner: to be
  assigned. `Arg.Wmma_info` becomes `(dims, dtype_in, device, threads,
  tc_upcast_axes)`; the name and output dtype are derived
  (`WMMA_{dims}_{dtype_in.name}_{dtype.scalar().name}`, spaces to underscores)
  and `expand_wmma` gates on the `tc_upcast_axes` slot instead of the node tag.
  Starts in `lib/uop/uop.ml{,i}` and sweeps `renderer/cstyle.ml`, `device.ml`,
  `program_spec.ml`, `uop/spec.ml`, `codegen/opt/postrange.ml` and
  `codegen/codegen_lower.ml`.

  *What catches it*: `test/unit/test_cstyle.ml` pins the emitted primitive name
  at both the preamble and the call site for three backends —
  `__WMMA_8_16_16_half_float` (CUDA), `__WMMA_8_8_8_float_float` (Metal) and
  `WMMA_16_16_128_float8_e4m3_float` (AMD fp8) — so a derivation that disagrees
  with today's stored name desynchronises the `#define` from its caller and
  goes red. `test/unit/codegen/test_postrange.ml:800` asserts
  `node_tag result <> None`, which pins the *old* gate: it must be rewritten as
  part of the change, and its going red is the signal that the gate moved.

  What `renderer/cstyle.ml` needs specifically: `wmma_args` returns
  `(name, dims, dtype_in, dtype_out, device, threads, upcast_sizes)` where
  `upcast_sizes` is read off `src[i].shape[-1]` rather than folded from
  `upcast_axes`; `amd_wmma_prefix`, `cuda_wmma_helpers` and the Metal preamble
  take the derived name; and the `Ops.Wmma` string rule uses it in place of
  `v.info.name`.

- **3. `ParamArg.multiple_of`** (`980748ccf`). Owner: to be assigned. A
  variable declared as a multiple of *k* lets `fold_divmod_general` fold
  `x % k` to zero and refuse to split `x / k`. The field lands in
  `lib/uop/uop.ml{,i}`; readers are `uop/divandmod.ml`,
  `schedule/rangeify.ml`'s `to_define_global`, `schedule/multi.ml`'s
  `param_to_multi`, `callify.ml`'s input-buffer replacement, the `bind`
  divisibility assertion, and `device.ml`'s `add_param_arg` cache key.

  *What catches it*: **nothing.** `multiple_of` has zero occurrences in `lib/`
  and `test/` today, and nothing constructs a variable with a non-unit
  multiple, so every threading site could drop the field on the floor and every
  suite would stay green. Write the `test/unit/codegen/test_divandmod.ml` case
  first — a `PARAM` with `multiple_of = 4` whose `% 4` folds to zero and whose
  `/ 4` is left alone — then thread the field.

- **4. `Uop.program_info.aux` and `program_info_from_sink ?aux`.** Owner:
  team-lead (`lib/uop/uop.ml` was locked during the pin update). Every other
  site is already deleted; these two are all that remain, plus the
  `aux = []` placeholder they let `program_spec.ml`'s `program_info` drop.

  *What catches it*: the field is constructed in
  `test/unit/uop/test_uop.ml` (including a test asserting it participates in
  `semantic_key`) and in `test/unit/uop/test_serialize.ml`'s round-trip, so
  both fail to compile until they are updated with it — a half-done deletion
  cannot land silently. `test/golden/debug` also loses the `aux` key from the
  `ProgramInfo` repr; that is expected churn for the pin regeneration.

## Rendered-output movement pending regeneration

Everything below is a deliberate change from this pin update whose only visible
effect is that a golden's `.actual` no longer matches its `.expected`. They are
listed so the regeneration is read as intended movement rather than regression.

- Every `cuda_*` kernel gains `typedef unsigned int uint;` as its first prefix
  line, and CUDA and Metal both spell `uint32` as `uint`.
- A register-file buffer read more than once is bound to a named temporary
  instead of having the read re-emitted at each use.
- Scalar kernel parameters wider than int32 render `const long` /
  `constant long&` instead of a bare `long`; the Clang fixed-ABI wrapper
  narrows them out of their `long long` slot instead of raising.
- `ProgramInfo` reprs lose the `aux` key (`test/golden/debug`).
- Every threefry call loses two `& 0xFFFFFFFF` masks; the narrowing cast
  already discards the high half.
- `pow(int, float)` loses its round-and-cast and keeps the promoted float.
- `arange` over a range wider than int32 renders at 64 bits.
- `var` accumulates at `sum_acc_dtype` and computes its denominator on the
  host rather than as a tensor `relu`.
- `cummax`, `logcumsumexp` and `scaled_dot_product_attention` build their masks
  at bool instead of promoting a float `ones` through the comparison.
- `logcumsumexp` and `multinomial` each lose one explicit `expand`.
- Integer division without a rounding mode enters the float domain by an
  explicit cast rather than by the reciprocal's implicit promotion, which also
  fixes the intermediate node dtype in `mean` and `var` on integer inputs.
- `test/golden/cstyle` still needs the directory move recorded under T11; it
  has not been touched.

## Confirmed no-ops in the pin update

Recorded so nobody re-derives them from the upstream diff.

- **`lib/nn/**` has nothing to port.** The three upstream `nn/` hunks are
  `BatchNorm.calc_stats` dropping an `expand` (tolk has no `BatchNorm`),
  `_embedding_bwd`'s `multi` → `unshard` and `dtypes.index` → `dtypes.weakint`
  (tolk has no autodiff; `lib/nn/embedding.ml` is a one-hot matmul with no
  custom backward kernel), and a weak-dtype guard in `safe_save`
  (`lib/nn/state.mli` exposes `safe_load` and `load_state_dict` only).
- **`lib/device.ml` needs no change for 64-bit variables.** `prog.call`
  already takes `vals : int64 array`, so widening a kernel variable is a
  renderer-side change only. The `TinyELF`/`Program` restructure and
  `DepsTracker` belong to runtimes tolk does not port.
- **The `.val` sweep is a no-op in `lib/engine/`.** tolk reads constant
  payloads through `Const.view` and `Uop.const_int_value`, never through a
  node's `arg`, so `x.arg` → `x.val` has no tolk site. Checked, not assumed.
- **`optimize_local_size` already has the shape upstream changed it to**: it
  takes one loaded `Device.prog` and varies only the launch dims across
  candidates, rather than reloading the module per candidate.
- **`Elementwise.where` needed no coercion removal beyond the cast.** Every
  internal caller already passes a boolean: masks come from comparisons,
  `one_hot_along_dim`, `logical_not`, or `bitwise_or` of those.
- **`lib/uop/hashcons.ml` has nothing to port** — it is vendored from
  Filliâtre's hashcons library and has no upstream counterpart, so it is a
  no-op rather than a decline. The wave plan lists it beside `uop.ml`, which
  is only true in the sense that `_rebuild_dtype` interacts with hash-consing;
  that logic lives in `uop.ml`.

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
- `simplify_valid` had **zero** coverage, not merely no ordering coverage. The
  earlier note here said inverting the `_valid_priority` comparator left
  everything green and concluded the corpus does not discriminate on clause
  order. Instrumenting the function at entry showed why: 16 calls across the 70
  parity cases, 24 across the 82 codegen goldens, and **0** from
  `test_symbolic.ml` — and every one of them returned early on the guard that
  skips predicates containing an `Ops.Index`. The comparator, the sort, and the
  clause fold never executed at all, so inverting the comparator could not have
  failed anything.

  There are **three** distinct ways to call this function and prove nothing,
  and the corpus used all of them:

  1. **The guard.** A predicate whose slice contains an `Ops.Index` returns
     before anything runs. This is every parity and golden call.
  2. **The wrong harness.** `pm_simplify_valid` is composed into `sym`, not
     into `symbolic`, and `Symbolic.simplify` runs `symbolic` — so *no test
     driven through `Symbolic.simplify` can ever reach this code*, however
     carefully its predicate is built. This is a property of the matcher
     composition, not of the tests, and it is the trap to warn about: the
     obvious way to write a symbolic test is against `Symbolic.simplify`.
     **A test for this path must drive `Symbolic.sym` (or apply
     `Symbolic.pm_simplify_valid` directly).**
  3. **Trivial data.** `pm_simplify_valid` matches `Ops.And` with no dtype
     constraint, so an *integer* bitwise AND reaches it too — as
     `test_symbolic.ml`'s `masked_div` group does with `(x & -4) // 4`. There
     `parse_valid` fails on both operands, every priority is 0, the sort is a
     no-op, and the fold returns `None`. The call happens; the branches are all
     trivial.

  The general lesson, which is why the entry is written out rather than just
  fixed: reaching the code is not coverage if the data makes every branch
  trivial. Verify by perturbing the thing under test — invert the comparator
  and watch the test fail — not by reading the test and judging that it looks
  like it exercises the path.

  Closed by the `simplify_valid` group in `test/unit/uop/test_symbolic.ml`: a
  two-clause non-indexing predicate whose clauses have distinct priorities and
  whose result differs under the two orders, asserted both structurally and
  pointwise over the leaf domain, plus a companion applying
  `pm_simplify_valid` on its own so the reduction is attributable to this
  rewrite rather than to something else in `sym`.

  Verified by perturbation, not inspection: inverting the `_valid_priority`
  comparator to `Int.compare b a` turns both tests red (and nothing else in the
  uop suite), and restoring it turns them green. So the group does reach the
  comparator, and the comparator's direction is what it pins.

## Misc

- `Uop.program_vals` raises bare `Not_found` for an unbound variable; the
  reference raises a descriptive error naming the variable and its user.

- **`Ops.Wait` is deliberately retained with no constructor** (`f41e4a758`).
  That commit deletes `UOp.wait`, the WAIT arm of `dtype_from_uop`, and the
  `spec.py` rule, but leaves `WAIT` in the `Ops` enum, so tolk does too. The
  constructor therefore looks dead and is not: `test/unit/uop/test_ops.ml`
  compares `Ops.Group.all` against the live tinygrad enum member-for-member,
  so removing it turns that suite red. Delete it only when upstream does.

- **`Axis_type.Loop` names a different concept than it did before the
  2026-08 pin.** The old `Loop` — a counted software loop — is now `Weak`;
  the current `Loop` is the unbounded wait-loop (void-dtype `Range`, closed
  by a conditional `End`). Nothing constructs one yet. Two consequences.
  `Axis_type.to_string Loop` is again `"loop"`, the same atom the old kind
  fed into the `Diskcache` program key, so `cache_version` was bumped to 3
  to stop a pre-rename entry being found under the new meaning. And the
  parity/golden Python drivers still say `AxisType.LOOP` where they mean
  `AxisType.WEAK` — see the wave-9 precondition below.

  **Do a rename-and-reuse in two steps, and this is the evidence why.** The
  rename (`Loop` -> `Weak`) landed first, alone, so the compiler pointed at
  every reference. Adding the new `Loop` came second. Had both landed
  together every existing `Axis_type.Loop` would have kept compiling while
  silently meaning the new concept. That is not hypothetical: three
  assertions in `test/unit/uop/test_uop.ml` (1417, 1424, 1535) pin the axis
  kind as the *string* `AxisType.LOOP`, and one-step would have left them
  green while asserting the wrong class. Strings, cache keys and Python
  drivers are all outside the compiler's reach; only the ordering exposed
  them.

- **BLOCKING PRECONDITION FOR WAVE 9 — migrate nine `AxisType.LOOP` sites
  before regenerating anything.** These nine Python constructions mean the
  *counted* loop and must become `AxisType.WEAK`:

  | file | lines |
  |---|---|
  | `test/golden/cstyle/generate_expected.py` | 120, 200, 201 |
  | `test/parity/nested_loops/main.py` | 19, 20 |
  | `test/parity/token_gather_collapse/main.py` | 30, 31, 32 |
  | `test/parity/loop/main.py` | 18 |

  Both constructors exist at the new pin, so nothing raises: the drivers run
  and emit expectations for the wrong axis class. Regenerating first bakes
  that into every `.expected` and destroys the signal that would reveal it.
  Migrate, then regenerate — not the other way round.

- **There are two Ops-order tests and only one self-heals.**
  `test/unit/uop/test_ops.ml` probes the live `_tinygrad` checkout, so it goes
  green by itself when the pin moves. `test/unit/uop/test_uop.ml:70` holds a
  *hardcoded* 82-entry name list and must be edited by hand on any enum
  change. Until the pin moves, `test_ops.ml` is red on exactly one entry
  (index 78, `UNSHARD` vs the old pin's `MULTI`) — expected, not a
  regression, and not something to debug. The trap is that a stale hardcoded
  list fails identically and looks like the same known failure. On any enum
  change, diff the whole list against the reference rather than patching the
  entry you know about.
