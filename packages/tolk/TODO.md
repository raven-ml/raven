# TODO

Known gaps, deferred parity items, and follow-ups. Maintainer notes — reference
anchors point at the tinygrad clone.

`_tinygrad` is at `baa614806`, and the whole `.expected` corpus was regenerated
against it. `_tinygrad_next` is a leftover worktree at the same commit, kept
only so the pre-move corpus can still be reconstructed; it has no references
anywhere and can be removed once this round is reviewed.

Corpus status after the regeneration: **631 files, 565 agree with tolk, 66
diverge**, and every one of the 66 is attributed to one of four causes — see
"Residual divergences after the pin move" below. Nothing is red for pin drift
any more, so a red file now means a real gap.

## Open bugs

- **`Coalesce: multiple stores to the same offset` under the CUDA renderer
  only**: a cross-render probe found kernels that render cleanly on CPU and
  fail in the coalescer when the same kernel ASTs are lowered through the CUDA
  renderer. Not yet minimised.

- **FIXED — the CUDA WMMA primitive no longer renders with scalar operands.**
  It used to emit `float __WMMA_8_16_16_half_float(half a, half b, float c)`
  with empty asm operand lists; it now emits
  `float4 __WMMA_8_16_16_half_float(half8 a, half4 b, float4 c)`. Root cause
  was `cuda_wmma_helpers` reconstructing the operand widths from
  `info.tc_upcast_axes`, which `expand_wmma` clears to `None` as its
  already-expanded marker, so the `| None -> ([], [], [])` fallback made every
  width 1. Verified across all four tensor-core cases — `tc_matmul_32`,
  `tc_matmul_f16`, `tc_matmul_bf16`, `tc_matmul_fp8`, 16 files, all agreeing
  with the reference at `baa614806`.

## Deferred parity divergences — blocked

Each of these depends on something that does not exist yet. The blocker is
named; do not pick one up before it lands.

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

  **Blocker corrected**: it is not `Axis_type.Device`, which exists. The
  producer is `lib/schedule/multi.ml:84,101`, which still mints `_device_num`
  via `U.variable` instead of a DEVICE range — scheduler work, not a codegen
  gap. Porting it deletes tolk's `_device_num` special cases at
  `rangeify.ml:632,2011` and `indexing.ml:604`, but touches four files across
  three ownerships and moves twelve `multi_*` goldens, so it wants one commit
  by one owner.

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

## AMD runtime — deferred (KFD path)

The KFD-backed AMD device (`lib/runtime/amd`, wired as the `AMD` backend)
covers compile, dispatch, DMA host transfers, and fault reporting on
single-die parts. Deliberately not ported with it:

- **HCQ graph support** (`runtime/graph/hcq.py`): batched replay of a call
  sequence over the hardware queues. The device registers no `?graph`
  capability, so the engine falls back to per-call dispatch.

- **AQL queues / multi-XCC dispatch** (the `is_aql` path of
  `AMDDevice.__init__`, `AMDComputeAQLQueue`): `Tolk_amd.create` rejects
  parts with more than one compute die. The PM4 queue builders are already
  multi-die aware (`pred_exec`, per-die scratch slices); the AQL descriptor
  path is not built.

- **SQTT and PMC profiling** (`sqtt_start`/`pmc_start` in `ops_amd.py`):
  thread-trace and performance-counter capture. `sqtt_enabled` stays
  `false` and `Compute_queue.exec` rejects it loudly.

- **USB and remote device interfaces** (`USBIface`, the remote backend):
  only the kernel-driver interface is wired here; the driver-less PCI tier
  is its own workstream under `lib/runtime/amd/am`.

- **A committed hsaco fixture** for `Program.load` against real compiler
  output — generating one needs a Linux/ROCm machine. The hand-built ELF
  fixture in `test/unit/test_runtime_amd.ml` covers the loader meanwhile.

- **Real-hardware validation**: everything below the device layer is
  exercised by unit tests over mapped memory, and the device-level tests
  skip without `/dev/kfd`; nothing has run on a live GPU yet.

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
  `multiple_of` is an `int option` — `None` means no promise, not `1`. That
  reading is right for `Uop.param` and wrong for `Uop.variable`, which the
  reference defaults to `1`; see the residual-divergence section.

## Residual divergences after the pin move

The corpus was regenerated against `baa614806`. 565 of 631 files now agree with
tolk; the 66 that do not fall into exactly four causes, none of them pin drift.
Ordered by how much they matter.

- **The reciprocal of an `rsqrt` sits on the wrong side of a kernel boundary**
  (28 files: every `llama_*` case in `golden/codegen` and `golden/rangeify`,
  all four backends; plus 3 more since the AMD backend joined the corpus —
  `amd_llama_rmsnorm`, `amd_llama_ffn_gate`, `amd_llama_vector_scale` in
  `golden/codegen`, same cause, while `amd_llama_embedding` and
  `amd_llama_output_projection` agree). The reference stores `sqrt(s)` and
  divides at the consumer; tolk stores `1.0f/sqrt(s)` and multiplies:

  | | reference | tolk |
  |---|---|---|
  | `llama_rmsnorm` | `__builtin_sqrtf(s)` | `(1.0f/__builtin_sqrtf(s))` |
  | `llama_vector_scale` | `(val1[i]/val0)*val2[i]` | `val1[i]*val0*val2[i]` |
  | `llama_ffn_gate` | `(...)/val2` | `(...)*val2` |

  **Not a miscompile**: the two are consistent pairs and agree to float
  rounding. But the intermediate buffer genuinely holds a different value —
  the root on one side, its reciprocal on the other — so anything that reads
  that buffer outside these kernels would disagree. `RMSNorm` itself is
  identical at both pins and `rsqrt`'s definition did not change, so the owner
  is downstream folding (`mixin/op.py` and `codegen/simplify.py` are the two
  files that moved between the pins). Unported upstream change; find the rule
  and port it.

- **`multi_*` emits about half the kernels the reference does** (9 files:
  `multi_allreduce_naive`, `multi_allreduce_ring`, `multi_replicate_elementwise`
  at stage 5 CUDA and stage 7 CPU/CUDA). Counts are reference-vs-tolk 6/2,
  34/18 and 9/5 — the reference splits per-shard copies into their own kernels
  where tolk fuses them. This is the MULTI -> UNSHARD sharding rework, still
  listed as not started, now showing as a clean structural divergence instead
  of pin noise. Same area as the three failing `test_pmap.ml` cases.

- **`ParamArg` reprs omit `multiple_of`** (22 files, every stage-5 dump with an
  ALU-space parameter, plus both `golden/debug` files). The reference's
  `UOp.variable` defaults `multiple_of=1` (`uop/ops.py:983`); tolk's
  `Uop.variable` (`lib/uop/uop.ml:720`) takes `?multiple_of` and leaves it
  `None`.

  **Proven inert, and it is a one-line fix.** Both consumers behave identically
  under `Some 1` and `None`: `const_factor` returns `1` either way (the
  reference's PARAM branch yields `multiple_of`, tolk falls through to its
  `| _ -> 1`), and `divides v` for `v > 1` returns `None` either way (`1 % v`
  is non-zero, so the reference's branch fails; tolk never enters it). Give
  `Uop.variable` a `?(multiple_of = 1)` default and all 22 files go green.
  `Uop.param` must keep defaulting to `None`, which is what the reference does
  for non-variable params.

- **Sharded axis ids are off by one** (7 files, `multi_*` stage 5/7 CPU). The
  reference numbers the sharded range axis 1 and renders `Lidx1`; tolk numbers
  it axis 0 and renders `Lidx0`. Consistent with the reference reserving axis 0
  for the `AxisType.DEVICE` range that tolk does not build yet — the blocked
  `pm_device_to_var` item. Cosmetic today; it resolves with that work.

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

- **`c9e11544d`'s remaining cast deletions, decomp half.** Two arms landed in
  `codegen/simplify.ml`; five still carry their casts and belong to the decomp
  owner: `decomp_dtype.ml:180` (`L2i_add` carry), `:189` (`L2i_sub` borrow),
  `:560` (float→long `adj`), `:839` (`rne` sticky), `decomp_op.ml:371`
  (`floordiv_to_idiv` fixup), `decomp_transcendental.ml:456` (`xexp2`). Check
  each against a live spec rather than porting as cleanup: one arm was found
  load-bearing in the wrong direction, and another blocked a fold outright.
  The commit's other two arms need nothing — `frontend/rand.ml:105` is already
  cast-free and tolk has no ONNX frontend.

### Silent-default audit (2026-08-07)

A sweep of `lib/` for one bug shape: **tolk reconstructs a value that tinygrad
reads off the node, and the reconstruction carries a fallback for a case it
does not model.** The fallback yields a plausible number, so the failure is a
wrong answer rather than a diff or a crash.

**83 sites judged: 1 BUG (fixed), 11 LATENT (2 hardened), 71 CORRECT.** The
correct ones are almost all defaultdict equivalents (adjacency lists, degree
tables, name counters, tag side-tables, dispatch buckets), env-var defaults,
and probe failures where the reference defaults the same way. Each was checked
against its upstream counterpart.

**BUG — `codegen/late/gater.ml` `vzero_like`, fixed.** The value a gated LOAD
falls back to was built by re-deriving the access width from the *index
expression*, defaulting to one lane. An image coordinate addresses four floats,
so a gated image load zeroed one lane and left three unwritten; a
multi-dimensional shrink was undersized too, since `vmax` of a stacked size is
the largest dimension, not their product. The reference builds it from the load
itself (`l.vconst_like(0)`), whose width goes through `.shape` and raises.
Fix: `vzero_like` drops its `mop` parameter and reads the width off the node.
`Uop.max_numel` is now a first-class operation (`prod (max_shape u)`, raising
for a shapeless node), which is where tinygrad keeps it; `decomp_dtype.ml`'s
and `program_spec.ml`'s private copies are folded into it. Two remain:
`cstyle.ml:309`, and `validate.ml:26` which returns an option on purpose so the
OOB validator can skip a shapeless buffer. Pinned by "gater zeroes every lane
of a gated image load" (`test/unit/codegen/test_lower.ml`), which fails
`Expected: 4 / Actual: 1` against the old code.

**LATENT — unreachable today, nothing enforcing it.** Ordered by consequence.

1. `uop.ml` `compute_shape_opt`, broadcastable arm — `List.filter_map
   shape_opt` *drops* shapeless operands, then broadcasts the rest with
   `lenient_broadcast_shape`, which keeps "the first candidate" when two
   dimensions disagree. The reference asserts both away. An ALU node built
   without pre-broadcasting takes its first operand's shape, and every lane
   count derived downstream is silently wrong. A raising `broadcast_shape`
   already sits next to it. **This is the same defect as the rangeify
   `filter_map` bug, in the core shape rule.**
2. `uop.ml` `compute_shape_opt` final `| _ -> None` — the reference raises
   `NotImplementedError` for an op with no shape rule. A new op becomes
   silently shapeless instead of failing.
3. `program_spec.ml` `max_numel` — *hardened*; was `| None -> 1` feeding
   `Estimates`, which gates BEAM candidate pruning, so an undercount both
   survived pruning and lowered the bar for everything after it.
4. `linearizer.ml:22` `range_size` — `| None -> 1` multiplied into `run_count`.
   **Proven unreachable**: `U.ranges` admits only `Ops.Range`, and every such
   node comes from `Uop.range`, which always supplies `Arg.Range_info`.
5. `linearizer.ml:32` PARAM slot — *hardened*; `as_param` additionally demands
   exactly one src, so a PARAM that grew one would sort as slot 0 and reorder
   the emitted parameter declarations.
6. `linearizer.ml:248` `chain` — silently skips an END whose range is not a
   single RANGE, dropping a control-flow ordering edge. Safe only because
   `pm_split_ends` runs earlier in the same rewrite.
7. `linearizer.ml:283` `range_key` — sorts a non-RANGE as axis 0. Same
   unreachability argument as (4).
8. `codegen_lower.ml:624` `clone_ranges` — `| None -> r` returns the original
   as its own clone, so the substitution is `(r, r)` and two merged END groups
   share one range. Same shape as the accumulator-slot collision fixed today.
9. `callify.ml:178` Permute — a missing `Arg.Ints` yields `Some []`, so a
   permuted tensor reports a *scalar* shape.
10. `callify.ml:560` `pm_finalize` — `original_shape = []` for a
    symbolic-shaped tagged node shrinks the buffer to rank 0. `require_shrink`
    catches it later, so this is a delayed loud failure rather than a wrong
    value.
11. `coalesce.ml:206` `host_is_osx` — `with _ -> false` swallows a failed
    `uname -s` into "not Darwin", so `image_valid_dims` checks the wrong pitch
    alignment on the platform this project primarily targets. Also forks a
    process per call.

**Excluded files, judged read-only (24 sites in `lib/schedule/**` and
`lib/renderer/cstyle.ml`, owned elsewhere at audit time).** Most are
`Not_found -> 0/[]/""` on renderer side-tables and are correct. Five to hand to
their owners, in order of consequence:

- `schedule/indexing.ml:503` — `Option.value ~default:[] (shape_exprs x)` feeds
  range-ending and index generation, so a node whose shape cannot be
  reconstructed is processed as rank 0. `shape_expr_of` already has a
  `fallback ()` for that case, so the `[]` may be standing in for something
  that has a real answer. **The one to look at hardest.**
- `cstyle.ml:309` `max_numel` — `| None -> stack_count u`, already a partial
  repair, but a shapeless *non-Stack* node still gets one lane where `.shape`
  raises. `Uop.max_numel` (raising) now exists; ask whether the `None` arm can
  be asserted away, making it the same shape as the WMMA width fix.
- `cstyle.ml:1954` `wmma_operand_width` — the `[] -> 1` arm is right for a
  rank-0 operand; the `None -> 1` arm is the shapeless guess again. Reference
  is `src[2].shape[-1]`, which raises.
- `rangeify.ml:1146` — `match U.as_range u with Some v -> v.axis | None -> 0`
  as a sort key. Same shape as LATENT (7).
- `cstyle.ml:118` `online_cpu_count` — `with _ -> 1`, so a failed `getconf`
  silently means one CPU and kernels go single-threaded with no signal.
  Performance, not correctness.

Cleared on inspection: `allreduce.ml:102` (the default is taken only when the
element count is odd, where 1 is the exactly correct chunk alignment, not a
guess) and the side-table defaults at `indexing.ml:376,496` and
`rangeify.ml:1661,1731,1782`.

**General finding.** A correct guard *downstream* of a silent default is
worthless: the allreduce rule's `U.vmax` backstop would have caught the
rangeify `filter_map` bug had the filter ever let a `None` through. Never
classify a site as correct because something later checks it.

**Two corrections to the LATENT list above**, found while cross-reading this
file:

- (6) `linearizer.ml:248` `chain` skipping an END whose ended child is not a
  single RANGE is not an unguarded hazard — it is the tolk-side counterpart of
  the guard the reference puts on `simplify_merge_adjacent`, and belongs to the
  void-RANGE loop header in DIVERGENCES.md. Leave it until that
  feature acquires a producer.
- `uop.ml` `ended_ranges` (and its copy in `codegen_lower.ml:505`) returning
  `[]` where `_tinygrad_next` returns `src[1:]` for UNSHARD is not a new
  finding either: it is downstream of the sorted-axis-tuple rework named as a
  blocker under "Deferred parity divergences — blocked". `Ops.Copy` staying in
  `range_start` is likewise consistent with COPY still taking part in rangeify.
  Worth knowing that the `[]` fallback is what makes both invisible: when the
  axis-tuple rework lands, this is the arm that has to move with it, and
  nothing will fail if it is forgotten.

### Silent-default audit, second pass: the structural variant (2026-08-07)

The fallback need not be a literal. `List.filter_map` over an all-or-nothing
derivation *drops* the failures and computes from the survivors — the same
defect, with no `| None ->` to grep for. Swept every `List.filter_map` (50),
`List.concat_map` (41) and accumulating `fold_left` (25) in `lib/`: **no new
BUG**. The variant exists in exactly two places, and both reconstruct a shape
from operands — `rangeify.compute_shape_of` (below) and LATENT (1) above.
Everything else is selection over legitimate non-participants.

Two sites do the idiom correctly, and are the shape to look for when reviewing
a `filter_map` over a derivation: `uop.ml:2925` pairs the filter with
`if List.length decomposed <> List.length xs then <bail>`, and
`codegen_lower.ml:244` `expand_broadcast` leads with `if List.exists
Option.is_none shapes then None`.

Recorded as a deliberate non-finding: `gpudims.ml:225` filters `local_idxs` to
SPECIALs and then zips the result *positionally* against `global_max`, so a
dropped entry would shift the alignment. The reference does character-for-
character the same thing (`[_dim_max(u.src[0]) for u in local_idxs if u.op is
Ops.SPECIAL]`, then `zip`). If the shift is a hazard it is upstream's.

**`schedule/rangeify.ml` `compute_shape_of` / `compute_shape_expr_of` — fixed,
and the answer was not the obvious one.**

*Is the reference strict here?* Yes, by not having the function. tinygrad's
`schedule/rangeify.py` reads `x.shape` off the node throughout — the raising
property. There is no `shape_of` in it. tolk's `shape_of`/`shape_expr_of` are a
*third* parallel shape derivation alongside `Uop.shape_opt` and
`callify.compute_shapes`, each with its own broadcast function and its own
leniency policy. That duplication is the standing design debt; see the
"Reconstructed values" bullet below.

*What making the filter strict broke.* Nothing visible, which was the problem.
Instrumented, the strict branch fires **308 times** per golden-corpus run (188
MUL, 82 ADD, 34 MAX, 4 SUB) in `shape_of` and never in `shape_expr_of` — and in
every single case the dropped operand is an `Ops.Load`. Goldens stayed
byte-identical and every suite passed anyway, so 308 answers flip from
confidently-right to `None` and the corpus cannot tell. That is exactly the
coverage situation that let the original miscompile through, so strictness on
its own would have been a change made blind.

*What had to be fixed alongside.* `compute_shape_of` ended with a bare
`| _ -> None`, while its own sibling `compute_shape_expr_of` ends with a
`U.shape` backstop. So a LOAD — whose shape `Uop.shape` knows perfectly well —
came back unknown, and the `filter_map` had been papering over that omission
308 times a run. It got the right answer by luck: dropping a LOAD and taking
the other operand's shape is correct exactly while the two agree, which is what
the elementwise contract says right up until it does not. With the same
backstop added to `compute_shape_of`, the strict branch **provably never fires**
— re-measured to zero hits over both golden corpora. That is what makes the
pair a local fix rather than a scoped change.

*Evidence.* `test/unit/{codegen,engine,frontend,nn,runtime}/runtest` green;
`test_schedule_rangeify` 67/67; `engine/test_multi` 10/10. `golden/cstyle` 0/62;
`golden/codegen` 12/82 and `golden/rangeify` 16/88 — the *identical file set* as
the baseline measured with the change reverted, all of them the recorded
rsqrt-reciprocal residual under "Residual divergences after the pin move", not
drift from this change.

*What is not pinned.* No test and no changelog entry: `shape_of` is not in
`rangeify.mli` and the change is output-neutral on everything covered, so there
is nothing observable to assert. The proof is the instrumented zero-hit
measurement, which does not survive as a regression test. The backstop also
widens `shape_of`'s answers on paths the corpus does not reach; `test_multi` is
the only multi-device coverage it has.

*The transferable lesson.* When a lenient site turns out to be load-bearing,
the fix is rarely to add strictness — it is to find the missing case the
leniency was standing in for. Strictness is then free, and you know it is free
because you can measure that the strict path no longer fires.


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

Both remaining entries need a clone-generated `.expected`, and neither can
produce a correct one today — see the two blockers named below.

- **`multi_stack` parity case** (the STACK sharding rule has no coverage).
  **Blocked on the pin move.** The reference's `multi_*` output is not stable
  across the pin: regenerating the existing multi cases at `baa614806` changes
  kernel signatures (`E_4_4n1(data0, data1, data2)` becomes
  `E_4_4n5(data0, data1)`), axis ids and kernel-name suffixes — those 14 files
  are the only ones in the whole corpus whose regeneration is semantic rather
  than cosmetic. An expectation captured now would be stale within a commit.

- **Image golden from the clone** (gates the image-coordinate convergence).
  **Blocked on the `lib/` convergence it is meant to gate.** tolk's
  `transform_to_image` emits the stacked `INDEX(buf, STACK[y,x])` form while
  the reference emits two-axis `INDEX(buf, y, x)`, so a golden generated from
  the clone today lands red on that known divergence rather than on anything
  it pins. Generate it in the same commit that converges the producer.

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

Closed since this list was written, recorded for what they pin:

- **WMMA FLOP factor** — `test/unit/test_program_spec.ml`, three cases in
  `Estimates.of_program`: the per-thread count `2*M*N*K/threads`, that the
  divisor is read off the node (same dims at 32 and 64 threads give 128 and
  64), and that a loop multiplier scales it. The factor was previously visible
  only through goldens, where a wrong estimate does not change rendered source.
- **Group-reduce gate-mask shape** — `test/unit/codegen/test_gpudims.ml`,
  "two missing local ranges gate on a bool AND of equalities". Two missing
  locals are required: one cannot distinguish a fold from a nest. It also
  settles the concern that prompted the entry — there is no divergence.
  `gate_missing_locals` folds `Cmpeq(range, 0)` left under `Ops.And`, and the
  reference's `UOp.uprod` reduces with `operator.and_` when the operands are
  bool (`mixin/elementwise.py:37`), so the two are the same construction.
- **`bf16_vector_load_reindexes_shrink`** — verified end-to-end and tightened.
  The old assertion used `List.exists` over `vmin = 3 || vmin = 4`, which a
  width-1 result would still satisfy; it now requires a 2-source STACK whose
  INDEX offsets are exactly `[3; 4]`, so losing the shrink's width fails.

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
  to stop a pre-rename entry being found under the new meaning.

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

- **The Python drivers are migrated to `baa614806` (done).** Before the
  migration 26 of 71 parity cases and 3 of 4 golden generators aborted against
  the new pin; after it, **all 71 parity cases and all 4 generators run, with
  zero aborts and zero skips** (2 + 62 + 88 + 82 golden files written). Five
  API breaks, none of them in tolk:

  | break | sites | fix applied |
  |---|---|---|
  | `UOp.const` swapped its arguments | 96 calls in 29 files | `UOp.const(dtypes.X, v)` -> `UOp.const(v, dtypes.X)` |
  | `UOp.const`'s `shape=` parameter deleted | 1 | `.expand(S)`, which is what the old body did |
  | `dtypes.index` deleted | 6 | `dtypes.weakint` |
  | `Invalid` is always bool | 2 | `UOp(Ops.CONST, dtypes.float32, (), Invalid)` -> `UOp.invalid()` |
  | `AxisType.WEAK` absent at the old pin | 9 | migrated earlier |

  Old signature `UOp.const(dtype, b, shape=None)`, new one
  `UOp.const(b, dtype=None)` (`uop/ops.py:624`). `dtypes.weakint` is not a
  guess: every one of the six sites has an OCaml sibling built with
  `U.const_int`, which is `const (Const.int Dtype.weakint n)`, and `UOp.range`
  now defaults to `dtype=dtypes.weakint`, so the constant must be weakint or a
  spurious cast appears against the range it is compared with. Likewise
  `Const.invalid` is already `{ dtype = Dtype.bool; ... }` on the tolk side,
  matching "Invalid is always bool, the promo lattice bottom"
  (`uop/ops.py:182`); the new `const` ignores any dtype passed alongside
  `Invalid`, so a float-typed one is now a spec violation.

  *What proved the swap did not silently transpose anything*: the 45 cases
  that already ran before the migration produce **268 of 268 `.expected` files
  byte-identical** afterwards. And the three cases carrying the judgement calls
  — `gated_store`, `token_gather_collapse` (4 of the 6 index sites),
  `rnn_grad` (the `shape=` site) — now match tolk's own `.actual` on 18 of 19
  files, the exception being the `multiple_of` movement below.

  **`golden/cstyle/generate_expected.py` hid all of this.** It catches per
  case and prints `SKIP <name>: <error>`, then `Done. Generated 0 .expected
  files`, and **exits 0**. All 62 of its cases were skipping. Read its output;
  do not trust its exit status.

- **The nine `AxisType.LOOP` driver sites are migrated to `AxisType.WEAK`
  (done).** Three in `test/golden/cstyle/generate_expected.py`, two in
  `test/parity/nested_loops/main.py`, three in
  `test/parity/token_gather_collapse/main.py`, one in
  `test/parity/loop/main.py` — all nine mean the *counted* loop, and all nine
  OCaml siblings already said `Axis_type.Weak`. No `.expected` was
  regenerated; that happens once, after the pin moves.

  **The pin has since moved and the corpus is regenerated**, so these drivers
  now match their clone. `_tinygrad` was fast-forwarded on `master`
  (7eb197b1b -> baa614806, a clean fast-forward, working tree still clean) and
  the untracked `.venv` survived — which matters, because `test_ops.ml` prefers
  it and it is Python 3.11 against a system 3.14. Every clone reference is a
  relative path to `_tinygrad`, so nothing needed a path edit, and nothing
  anywhere references `_tinygrad_next`.

- **There are two Ops-order tests and only one self-heals — but both are
  ready for this move.** `test/unit/uop/test_ops.ml` probes the live
  `_tinygrad` checkout, so it goes green by itself when the pin moves.
  `test/unit/uop/test_uop.ml:70` holds a *hardcoded* 82-entry name list and
  must be edited by hand on any enum change — that edit has already been made
  for this move: the list is byte-identical to `baa614806`'s enum, and differs
  from `7eb197b1b`'s on exactly one entry. So `test_ops.ml` is red today on
  that same single entry (`UNSHARD` vs the old pin's `MULTI`) — expected, not a
  regression, and not something to debug. The trap is that a stale hardcoded
  list fails identically and looks like the same known failure. On any enum
  change, diff the whole list against the reference rather than patching the
  entry you know about.
