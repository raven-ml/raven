# TODO

Known gaps, deferred parity items, and follow-ups. Maintainer notes — reference
anchors point at the tinygrad clone.

Two clones are live during the pin migration: `_tinygrad_next` @ `baa614806` is
the pin `lib/` is being ported to, and `_tinygrad` @ `7eb197b1b` is the pin the
committed `.expected` corpus was generated from. A byte-diff against the wrong
one reads convergence as divergence — see "Rendered-output movement pending
regeneration" below for what is expected to be red until wave 9 regenerates.

## Open bugs

- **`Coalesce: multiple stores to the same offset` under the CUDA renderer
  only**: a cross-render probe found kernels that render cleanly on CPU and
  fail in the coalescer when the same kernel ASTs are lowered through the CUDA
  renderer. Not yet minimised.

- **The CUDA WMMA primitive renders with scalar operands.** Every
  `__device__ __WMMA_*` prototype comes out as
  `float __WMMA_8_16_16_half_float(half a, half b, float c)` with empty asm
  operand lists (`"{%0}, {},"` / `: );`) instead of the reference's
  `float4 __WMMA_8_16_16_half_float(half8 a, half4 b, float4 c)`. The kernel
  body is byte-identical either way — it calls `make_half8(...)`/`make_half4(...)`
  correctly — so only the emitted primitive is wrong, and it is uncompilable.

  `cstyle.ml`'s `cuda_wmma_helpers` reconstructs the three operand widths from
  `info.tc_upcast_axes` (`| None -> ([], [], [])`, so `axis_product` is 1). But
  `expand_wmma` (`codegen/codegen_lower.ml:156`) clears that slot to `None` as
  its "already expanded" marker, so by render time it is *always* `None`. The
  reference reads the widths off the node instead —
  `wmma_args` is `tuple(uop.src[i].shape[-1] for i in range(3))`
  (`renderer/cstyle.py:113`). Fix at that owner: take the widths from the WMMA
  node's own three sources, and drop `tc_upcast_axes` from the `dedup_by_key`
  (it is a constant `None` there, so it deduplicates nothing). `device.ml:575`
  also mixes the always-`None` field into a key.

  Introduced by the `wmma_info` 5-tuple rework. Metal is unaffected because
  `metal_wmma_helpers` hardcodes `~sz:2` rather than reconstructing. This is a
  fifth instance of the reconstructed-value pattern under "Design debt".

  *What catches it*: `test/parity/tc_matmul_32` (new, see below) and the three
  `tc_matmul_{f16,bf16,fp8}` cases. `tc_matmul_32`'s `stage7_cuda` disagrees on
  exactly this and nothing else.

## Deferred parity divergences — blocked

Each of these depends on something that does not exist yet. The blocker is
named; do not pick one up before it lands.

- **Custom-kernel inputs are not force-realized** (`470c032a5`, `f253c4469`).
  Upstream's `realize_custom_kernel_srcs` realizes the arguments of a CALL
  whose body is a SINK or PROGRAM and marks them non-removable. tolk has no
  entry point that builds such a CALL in a tensor graph (no `custom_kernel`),
  so the rule and the `non_removable` set it feeds have no consumer.

  **Blocker**: the `custom_kernel` feature. Port both with it, not before.

- **The reference's UNSHARD, COPY and CALL spec rules are not in
  `lib/uop/spec.ml` yet.** `Ops.Unshard` is named but `Uop.unshard` still
  builds a single-axis `Arg.Int`, so the reference's
  `len(src) == 1+len(arg)` rule has nothing to check; COPY keeps
  `allow_any_len` until copies leave rangeify (`dcad11941`); and the
  address-valued `CALL` rule needs call-inside-C (`6e979b879`).

  **Blocker**: the sorted-axis-tuple rework for UNSHARD, the COPY move out of
  rangeify, and call-inside-C respectively. A spec that admits a shape nothing
  builds proves nothing — port each rule with its owning change.

  (The 5-tuple WMMA arg check is **done**: `spec.ml`'s
  `op ~src:[any; any; any] Ops.Wmma =?> Option.is_some (Uop.as_wmma u)` is the
  typed equivalent of the reference's `len(x.arg) == 5`.)

- **BITCAST/COPY fold extension** (callify): blocked on the same tag machinery
  as the bare-view aliasing deferral; the COPY arm additionally needs the
  disk-copy push rules tolk does not port.

- **Host I/O via allocator bridge** (`frontend/run.ml`): upload/download use
  `Buffer.copyin`/`as_bytes` directly because tolk has no host pseudo-device
  to route a `copy_from` through (reference: `device.py:113-115` seeds via a
  PYTHON-device buffer).

  **Blocker**: a host pseudo-device. Converges to `copy_from` if one lands.

- **`pm_device_to_var` is not in `codegen/gpudims.ml`** (`7d4892629`). The
  reference lowers an `AxisType.DEVICE` range to the `_device_num` variable at
  the end of `pm_add_gpudims`, and drops that range from the `END`s that closed
  it. `Axis_type.Device` and `Ops.Unshard` both exist now, but nothing
  constructs a DEVICE range, so there is nothing to lower.

  **Blocker**: the MULTI → UNSHARD rework, which is what starts building DEVICE
  ranges. Lands with it, together with the matching DEVICE guards in
  `codegen/simplify.ml`'s `mark_range_mod` and `opt/postrange.ml`'s `rngs`.

  *What catches a bad port*: `test/unit/engine/test_multi.ml` and the twelve
  `test/parity/multi_*` cases, which dump both the scheduled graph (stage 5)
  and the rendered source (stage 7) — a DEVICE range that reaches the opt axes
  or fails to become `_device_num` shows up in both.

- **`multi_pm` does not run at the head of `full_rewrite_to_sink`.** The
  reference resolves in-kernel shards (register fragments) during codegen, not
  only in the scheduler. Tolk's `multi_pm` needs scheduler-supplied shape and
  device callbacks and there is no in-kernel `Ops.Multi` to resolve, so the
  pass would be a no-op traversal.

  **Blocker**: UNSHARD and `alloc_fragment`. *What catches a bad port*:
  **nothing today** — with no in-kernel shard to resolve, adding the pass is
  unobservable either way. It becomes testable only once `alloc_fragment` can
  put a shard inside a kernel; that commit needs the test.

- **Negative slice bounds against a symbolic size** (`mixin/movement.py`). The
  reference resolves `start`/`stop` against a possibly symbolic `size` before
  deciding whether the slice is fully concrete. `Movement.parsed` is built from
  `Tensor.shape`, which raises on a symbolic dimension, so tolk cannot reach
  the case at all.

  **Blocker**: symbolic-shape `getitem`.

## Deferred parity divergences — not started

Nothing external blocks these; each could be picked up today.

- **COPY still takes part in rangeify** (`f65001e29`, `dcad11941`). Upstream
  converts every COPY to a BUFFER+STORE before rangeify (`pm_copy_to_store`),
  drops COPY from `ALWAYS_CONTIGUOUS`, from the realize map and from
  `ALWAYS_RUN_OPS`, lets `split_store` emit a plain kernel for it, and rebuilds
  the COPY afterwards in `schedule/__init__.py` (`pm_copy_from_store`,
  `copy_kernel_to_copy_uop`, `simplify_copy_kernel`) with
  `assert_all_same_devices` replacing the device check inside `split_store`.

  The old blocker is gone: `Simplify.simplify_ranges` and
  `Simplify.flatten_range` are both exported from `lib/codegen/simplify.mli`
  now. What remains is sequencing — the second half lands in
  `lib/engine/realize.ml` (copy detection) and it moves every rangeify golden,
  so it wants one commit that includes the golden regeneration.

- **The threefry symbolic rules are not pruned yet** (`33755a346`).
  `lib/uop/symbolic.ml` still carries five unpack rules (the `&0xFFFFFFFF`
  cast, both `*2^32` splice forms, and both `<<32` forms); the reference keeps
  only the two `<<32` ones. This was one third of a three-file change; two
  thirds have landed — `threefry2x32` in `lib/codegen/decomp/decomp_op.ml` now
  splits, rotates and repacks with `<<`/`>>` and narrows the low word by cast
  rather than mask, and `_threefry_random_bits` in `lib/frontend/rand.ml` no
  longer masks before its narrowing casts. All that remains is pruning the
  three now-dead symbolic rules.

- **Kernel formals keep their caller-side variable names** (`be25207a7`).
  Upstream renames a BIND'd scalar formal to `p{slot}` in `UOp.param_like` and
  maps it back in `resolve_linear_call` with a `substitute(..., enter_calls)`.
  tolk identifies kernel-body variables by name and address space
  (`engine/schedule.ml`, `variables_of_kernel_body`), so two callers that bind
  different values to the same variable name are not distinguished. Scoping
  hygiene, not a live bug — no tolk failure is known to depend on it.

- **`has_index` duplicates `op_in_backward_slice_with_self` because
  `backward_slice` is uncached.** Both feed `fold_where_closure` in
  `lib/uop/symbolic.ml`. The reference expresses that guard as
  `u.op_in_backward_slice_with_self(Ops.INDEX)`, which costs nothing there
  because `backward_slice` is a cached node property; tolk's
  (`lib/uop/uop.ml`) is a plain `toposort` filter and `in_backward_slice`
  re-toposorts per call, so routing a rule that runs on every `where` in every
  pass through it would be quadratic. Hence the local memo. Both collapse into
  the node API once `backward_slice` is cached — a change to a hot path that
  has to be measured against the current rewrite engine, so it is its own piece
  of work, not a rider on whichever wave notices it.

  (`bool_slice` itself has moved to `lib/uop/uop.ml`, where the reference keeps
  it, exposed as `bool_slice_mem` — a membership query rather than a table, so
  the memo stays private and no caller can mutate a cached set.)

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

  Not observable on the committed tensor-core cases: `test/parity/tc_matmul_32`
  is byte-identical to the reference on Metal at 32³ (four WMMAs), and the
  `tc_matmul_{f16,bf16,fp8}` cases have a single WMMA each. Verified identical
  under both `7eb197b1b` and `baa614806`.

- **Leading-dim view folds**: last-axis contiguous views fold to schedule-free
  aliases (matching the reference), but callify's local
  `contiguous_view_offset`/`make_slice` bails when a leading dim is not kept
  in full, so e.g. a 2-D row slice still materializes a copy. The exported
  `Uop.contiguous_view_offset` already handles leading dims — align the fold
  to it and delete the partial local reimplementation. Schedule-shape change
  (kernel counts drop), so verify against the golden suites.

- **expand_bitcast** (rangeify): a BITCAST between dtypes of different
  itemsize has to be re-expressed as a reshape, a per-part shift and a
  recombination. It blocks half-precision `Rand.rand`; no golden exercises it.

  Two `lib/uop/` movement primitives are missing and are part of this work, not
  a blocker on it: an axis-adding `stack ~dim` (tolk's `Uop.stack` builds a
  vector/shape tuple, not a new axis) and `flatten` over a negative axis range.
  `squeeze` is expressible as a `Uop.reshape`, and `Uop.shrink`/`cast`/`usum`/
  shift cover the rest — so the rule is roughly ten lines once those two land.
  Reference: `rangeify.py:116` plus `mixin/movement.py`.

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

- **`c9e11544d`'s remaining cast deletions each need checking against a live
  spec rather than porting as cleanup.** The left-bias blocker is gone:
  `alu_binary` now takes `promo_dtype` of its operands, so a cast that only
  existed to force the result dtype is genuinely redundant.

  **But "redundant" is not the only thing a cast can be.** The first arm
  ported under the new rule — `rule_reduce_and_where` in `codegen/simplify.ml`
  — turned out to be **load-bearing in the wrong direction**. It emitted
  `reduce * x.cast(c.dtype)`, and once `x` is correctly bool that cast is
  immediately re-consumed by `rule_mul_casted_bool` (`v * gate.cast` ->
  `gate.where(v, 0)`), so the two rules undid each other and the Mul never
  survived. Deleting the cast fixed a rule pair that had been silently
  cancelling. A deletion that reads as tidy-up can therefore change which
  *other* rules fire.

  So for each remaining arm — `codegen/simplify.ml` (`fold_result`,
  `rule_lift_add_lt`), `decomp/decomp_dtype.ml` (`l2i` ADD/SUB carry, `rne`),
  `decomp/decomp_op.ml` (`rule_floordiv_to_idiv`),
  `decomp/decomp_transcendental.ml` (`xexp2`) — check what else matches the
  cast before removing it, and land it with `SPEC=1` live.

  *What catches a bad port*: `Spec.type_verify`, which is now live —
  `test/unit/codegen/dune` sets `SPEC=1`, so a mistyped ALU node fails there
  instead of reaching the renderer and emitting plausible C. That suite is the
  one that runs lowering, so it is where the gate has to be; setting `SPEC=1`
  on `test/unit/uop` would be a no-op.

- **`cat` does not take the same-shape `stack` fast path** (`250de4b14`,
  final state `e684fcc68`). The reference's `stack` is a movement op
  (`_mop(Ops.STACK)`) and `cat` delegates to it when every input has the same
  extent on the cat axis. tolk inverts the relationship: `Op.stack` is
  `unsqueeze` + `Op.cat`, so the reference's fast path would recurse. Needs
  `Ops.Stack` as a tensor-level movement op first; `Uop.shape` already gives it
  the right shape (`(len src) :: shape src.(0)`), but nothing constructs one
  from the frontend. The `pm_reduce_collapse` half of `e684fcc68` is a
  separate, independent change.

## Deliberate divergences (do not port)

Reviewed and declined. Each would add machinery whose only producers are
runtimes or dependency orders tolk does not have. Do not re-open without a
consumer.

- **The void-RANGE loop header** (`b764599d8`, `39924387b`, plus its
  `spec.py` rules). The reference threads a "backedge" (the void range plus its
  bool condition) through `flatten_range` and `do_split_ends`, guards
  `simplify_merge_adjacent` against a non-RANGE ended child, filters void
  ranges out of `Scheduler.rngs`, and adds an `END(body, range, cond)` spec
  rule. The feature exists for HCQ wait loops — a runtime tolk does not port —
  and nothing in tolk constructs a void range, so all of it would be
  unreachable. `Axis_type.Loop` exists and names this concept, with no
  constructor; that is intentional. Port with a producer, never before.

  *What would catch a bad port*: **nothing, and nothing can** — there is no way
  to build a void range, so no test can reach the branches. The producer and
  the branches have to land together, with the producer's own test.

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
  with no reader. tolk's `aux` had no reader either: its only producer was
  `OpenCLRenderer.aux`, and nothing in `lib/runtime/` consumed it; it reached
  only the diskcache key and the debug repr. The whole channel is gone.
  `call_info.aux` is unrelated and stays; the reference keeps it.

- **The host-scalar `bitcast` stays in `symbolic.ml`** rather than moving to
  `dtype.ml` as the reference does (`67dc02d7e`). Forced, not skipped: tolk's
  equivalent (`bitcast_const_storage`) takes a `Const.t`, and `Const` depends on
  `Dtype`, so the move is impossible without re-cutting it to `storage_scalar`
  — for a consumer tolk does not have, since `lib/frontend/rand.ml` builds
  `_bits_to_rand` out of UOp `Bitcast` nodes rather than host-side arithmetic.

## Post-wave sweep — landed

All four changed a declaration in `lib/uop/` plus every reader across
`lib/codegen/`, `lib/renderer/`, `lib/schedule/` and the suites, so each went in
as one change rather than per-wave: the new `Axis_type.Loop` (added as its own
step, after the `Loop` -> `Weak` rename); the `wmma_info` 5-tuple
`(dims, dtype_in, device, threads, tc_upcast_axes)`; `ParamArg.multiple_of`; and
the deletion of `Uop.program_info.aux`.

Two invariants they created, which a later change could silently break:

- **`Tc.dtype_name` is the sole owner of the C spelling used in WMMA primitive
  names.** A second spelling anywhere desynchronises a `#define` from its call
  site, which is a link error rather than a diff.
- **`ParamArg.multiple_of` is in `device.ml`'s `add_param_arg` cache key.** Not
  optional: the field changes folding, so two programs differing only in it
  would otherwise collide and one would be handed the other's compiled code.
  `multiple_of` is an `int option` — `None` means no promise, not `1`.

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
- WMMA arg reprs become the 5-tuple `(dims, dtype_in, device, threads,
  tc_upcast_axes)` — the stored name, the output dtype and the always-empty
  reduce-axis list are gone, and the upcast axes now print as `None` once the
  operands have been contracted.
- `AxisType.LOOP` prints as `AxisType.WEAK` for every counted range: 61 lines
  across 27 `test/parity` `stage5_*.expected` files. The Python drivers must be
  migrated first — see the wave-9 precondition under Misc.
- Every threefry call loses two `& 0xFFFFFFFF` masks; the narrowing cast
  already discards the high half.
- `pow(int, float)` loses its round-and-cast and keeps the promoted float.
- `arange` over a range wider than int32 renders at 64 bits.
- `var` accumulates at `sum_acc_dtype` and computes its denominator on the
  host rather than as a tensor `relu`.
- `cummax`, `logcumsumexp` and `scaled_dot_product_attention` build their masks
  at bool instead of promoting a float `ones` through the comparison.
- `logcumsumexp` and `multinomial` each lose one explicit `expand`.
- Binary and ternary nodes no longer carry an `EXPAND` per operand from the
  frontend: promotion is dtype-only and `pm_expand_broadcast` widens during
  lowering. Any movement the scheduler used to fold away is simply never built.
- Scalar literals enter weak, so a literal paired with a narrow tensor no
  longer emits a cast to the default dtype and back.
- Integer division without a rounding mode enters the float domain by an
  explicit cast rather than by the reciprocal's implicit promotion, which also
  fixes the intermediate node dtype in `mean` and `var` on integer inputs.
- `test/golden/cstyle` still needs the directory move recorded under T11; it
  has not been touched.

## Confirmed no-ops in the pin update

Recorded so nobody re-derives them from the upstream diff.

- **Frontend promotion is dtype-only and its riders have landed**
  (`40f0d4af1`, `9433790ad`, `b1a72299a`, `b1060ca70`, `46b82d475`,
  `f19a2ad77`). `broadcasted` lives in `lib/frontend/elementwise.ml`, promotes
  to `Uop.promo_dtype`, rebuilds a weak constant weak instead of casting it,
  and leaves shapes to `pm_expand_broadcast` in codegen — a binary node's
  operands keep their own shapes and the node reports their broadcast. The
  `Tensor.broadcasted_hook` indirection and the `Op.broadcasted` re-export are
  gone with it, as are the two `Sys.opaque_identity Op.broadcasted` link
  anchors that existed only to run the hook installer. `Tensor.i`/`f`/`b` now
  build weak constants, which dissolved the `uf` helper that was triplicated
  across `elementwise.ml`/`op.ml`/`rand.ml`: a literal takes its peer's
  precision by construction rather than by inspecting it.

- **`_broadcast_to` computing its expand axes by hand is equivalent.** The
  reference uses `broadcast_axes(shape, new_shape)` (`f64f96ec5`), which tolk's
  uop layer does not export — equivalent for concrete shapes, which is all
  `Movement` accepts.

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
  of failing. Five instances:
  - `expr_numel` (`renderer/cstyle.ml`) consults the node's shape only for
    `Index`/`Shrink`/`Wmma` and otherwise propagates from sources, defaulting
    to one lane. The reference just uses `u.max_numel()`. `Wmma` was missing,
    so every WMMA declared scalar and tensor cores were uncompilable on all
    backends.
  - `cuda_wmma_helpers` (`renderer/cstyle.ml`) reconstructs the WMMA operand
    widths from `info.tc_upcast_axes`, defaulting to `([], [], [])` — i.e.
    width 1 — for the case the expander has already cleared. The reference
    reads `uop.src[i].shape[-1]`. Currently live; see "Open bugs".
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
    counter.

  The question this raises is whether these collapse into reading the node's
  own shape, or into doing what the reference does as it goes. Each is extra
  machinery relative to the reference, so it has to earn its place. Renderer-,
  scheduler- and codegen-wide, so it wants its own pass rather than a sixth
  point fix.
- **nativeint dispatch waist**: `Allocator.addr`/`prog.call`/`Graph.node`
  flatten buffers to `nativeint`, forcing the Metal token registry
  (`tolk_metal.ml`) and an `Obj.magic` in `Buffer.transfer`, and making
  multi-GPU CUDA transfer inexpressible (no device identities on the
  transfer). Plan: thread opaque `Buffer.t` (or a backend-downcast handle)
  through dispatch; runtimes bind (object, offset) themselves. Interim hard
  contract (enforced): the engine roots offset-view buffers across launch —
  the Metal registry is weak and a GC'd view raises at token resolve.

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
- **Trap to remember for `simplify_valid`-style rules** (the coverage gap
  itself is closed by the `simplify_valid` group in
  `test/unit/uop/test_symbolic.ml`): `pm_simplify_valid` is composed into
  `sym`, not into `symbolic`, and `Symbolic.simplify` runs `symbolic` — so no
  test driven through `Symbolic.simplify` can reach it, however carefully its
  predicate is built. A test for such a rule must drive `Symbolic.sym` or apply
  the matcher directly. Two more ways to call it and prove nothing: the guard
  that returns early on any predicate whose slice contains an `Ops.Index`
  (which is every parity and golden call), and trivial data — an *integer*
  bitwise AND reaches the rule too, and there every branch is a no-op.
  General lesson: reaching the code is not coverage if the data makes every
  branch trivial. Verify by perturbing the thing under test — invert the
  comparator and watch the test fail — not by reading the test and judging
  that it looks like it exercises the path.

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

- **The nine `AxisType.LOOP` driver sites are migrated to `AxisType.WEAK`
  (done).** Three in `test/golden/cstyle/generate_expected.py`, two in
  `test/parity/nested_loops/main.py`, three in
  `test/parity/token_gather_collapse/main.py`, one in
  `test/parity/loop/main.py` — all nine mean the *counted* loop, and all nine
  OCaml siblings already said `Axis_type.Weak`. No `.expected` was
  regenerated; that happens once, after the pin moves.

  **These drivers now require the new pin.** `AxisType.WEAK` does **not**
  exist at `7eb197b1b` — that pin's enum is
  `GLOBAL, WARP, LOCAL, LOOP, GROUP_REDUCE, REDUCE, UPCAST, UNROLL, THREAD,
  PLACEHOLDER`, with no `WEAK` member at all. So the four drivers raise
  `AttributeError` against the old clone and are correct only against
  `baa614806`. That is deliberate and harmless — nothing runs them until
  regeneration, and `dune runtest` diffs committed `.expected` against the
  OCaml `.actual`, never against Python. But **do not run `regen.py` or
  `generate_expected.py` until `_tinygrad` is at the new pin**; they will
  fail, not silently emit the wrong axis class.

  Regeneration must also repoint `test/parity/helpers.py` and
  `test/golden/codegen/generate_expected.py`, which both still `sys.path`
  in `_tinygrad`; once that clone is the new pin, no edit is needed beyond
  deleting the `_tinygrad_next` path if anything references it.
  `test/parity/tc_matmul_32` is the one case whose `.expected` was generated
  from `_tinygrad_next` already, so it should not move — verify it stays
  byte-identical rather than assuming it regenerated.

  The 61 `AxisType.LOOP` lines still in 27 committed
  `stage5_*.expected` files are *output*, not source. They are the pending
  movement recorded above and must not be hand-edited.

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

## Multi-device: three pmap cases fail after the pin migration

`packages/rune/test/test_pmap.ml` — `elementwise+matmul+reduce matches jit on
2 devices`, the same on 4, and `two collectively reduced outputs` — fail with
`Invalid_argument("buffer copy: size or dtype mismatch")` from
`engine/realize.ml:209`. That check is unmodified: a cross-device copy is
being scheduled between buffers of genuinely different sizes.

The sharding and allreduce half of the multi rework (`schedule/multi.ml`,
`schedule/allreduce.ml`) was never started, while the op it keys on was
renamed and operand broadcasting moved out of the frontend. So the token and
the surrounding shapes moved without the semantics that were meant to
accompany them.

One half is already fixed: sharding split a size-one axis, which is a
broadcast axis and must stay whole — the reference guards the same case in
`shard_srcs` ("broadcast srcs stay whole"). That took the failure count from
six to three and moved the rest past the divisibility check into this copy.

To resume: instrument `realize.ml:209` to print both buffer sizes, then work
back to which side is sharded and which is whole. The likely shape is a
consumer still treating a now-whole broadcast operand as if it were a shard.
The three tests are the acceptance criterion.
