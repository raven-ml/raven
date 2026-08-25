# Staged scan under jit — progress

Branch: `scan-jit`. Design notes: `packages/rune/doc/05-staged-scan.md`.

Goal: `Rune.jit (fun p -> ... Rune.scan ...)` (incl. `grad`) compiles the fold step once as a
loop in the compiled program instead of unrolling the whole trace.

## Status: COMPLETE — forward and reverse staged scans work, full raven suite green

`Rune.scan` under jit stages as a loop; `Rune.grad` of a scan under jit stages a reversed
loop reading the forward loop's carry stack. Multi-leaf carries, vector carries, nested
scans, captured weights, `n=1`, and replay all verified against eager execution. 51/51
rune.jit tests pass (8 scan tests), full rune + raven suites green except the 2
pre-existing environmental `tolk.uop.ops_parity` failures (need a vendored tinygrad
checkout; they fail identically on the base branch).

## What is done

**Rune**

- `lib/rune_scan.ml` (new): packed carry/step types, `E_scan` / `E_scan_bwd` effects,
  eager-fold fallback. `Rune.scan` attempts the effect; with no handler
  (`Effect.Unhandled _`) it falls back to today's eager loop — all non-jit paths unchanged.
- `lib/jit.ml`: tracer handles `E_scan` (`stage_scan`) and `E_scan_bwd`
  (`stage_scan_bwd`): traces the body once with placeholder slot tensors, schedules the
  body as its own compiled sub-linear (`schedule_body_linear`, captured unplanned), and
  emits a `CALL(CUSTOM_FUNCTION "loop")` node with slot descriptors. Multi-leaf carries
  are supported (per-leaf double-buffered carry pair, final buffer, carry stack).
  Forward loops always carry a per-leaf carry stack for the backward loop. `realize_arg`
  materializes device-less consts (e.g. a scalar carry init) into buffers.
- `lib/reverse.ml`: `E_scan` case = scan transpose: re-performs the effect (staged by an
  enclosing jit), records one tape entry that performs `E_scan_bwd` (captures the body's
  pullback against placeholder cotangents under a private tape). Eager fallback under a
  nested handler copy when no jit is present.
- `lib/forward.ml`, `lib/vmap.ml`: eager `E_scan` cases (nested handler copy); handlers
  made polymorphic-recursive where needed.

**Tolk**

- `lib/engine/realize.ml`: `exec_loop` — decodes the loop payload and replays the body's
  compiled linear once per iteration, re-seeding the in/out slot nodes (views at the
  per-iteration data index; reversed loops run `i = trip-1-j`), copying the carry stack
  and the init/final carry buffers. `run_linear` dispatches `CUSTOM_FUNCTION "loop"`
  through it (tag-level debug prints under `DEBUG>=2`). `dispatch_call` also routes
  `"loop"` calls, so nested loops (a scan inside a scan body) dispatch recursively.
- `lib/schedule/rangeify.ml`: `split_store` keeps a precompiled `CALL` as a store value
  whole — it replaces the STORE as the AFTER's kernel dep instead of building a kernel
  around it.

## Fixes landed this session (backward pass now computes correctly)

1. **Forward loop dead-code-eliminated under `grad`** (grad discards the loss value, the
   sole consumer of the loop's declared outputs, so the loop — including its carry-stack
   side effect — was dropped; the backward loop then read zeros). `stage_scan` now wraps
   each carry-stack buffer in an `AFTER` whose dep is the loop call and registers those
   in `scan_stacks`, so the backward loop's stack read is a real graph dependency.
2. **View offset units** in `exec_loop`'s `view` helper: the byte offset multiplied the
   element offset by `size` twice (invisible for scalar slots, crashes/misreads vector
   carries). Both call sites now pass element offsets.
3. **Loop-call argument layout helpers assumed an interleaved layout** (`1 + 3*i` etc.)
   while `args_nodes` builds a grouped one — identical for one leaf, wrong for
   multi-leaf carries. Fixed forward `pos_src/carry0/carry1/final/stack` and backward
   `pos_dc0/dc1/src` (the last was `4+3n+i` for a `3+4n+i` layout — it read the wrong
   leaf's final-carry cotangent).
4. **OCaml evaluation order**: per-leaf slot infos were collected by side effect inside
   `C.map` (whose constructor arguments evaluate right-to-left) but consumed in
   `C.iter` order (left-to-right), crossing leaf pairings for multi-leaf carries (and
   ppx_ptree records). Slots are now recovered by identity through `C.iter` passes over
   the slot structures (`slot_ins`/`dc_ins` tables).
5. **`scan_stacks` keyed on a plain `Hashtbl` of `(step, carry, xs)` object reprs**: a
   structural hash collision (nested scans stage two step records with the same closure
   code pointer) called polymorphic `compare` on a closure. Now identity-keyed on
   `Obj.repr step` alone via the existing `Tbl` (the step record is shared between the
   forward staging and the tape-recorded backward thunk).
6. **Buffer-slot collisions**: the scheduler allocates internal kernel buffers from a
   counter *local* to each schedule, seeded at the graph's max slot, while jit.ml's
   `make_node` uses the global `next_slot` — buffers created after a body scheduling
   (e.g. a loop's carry/dc buffers) could share a slot with the body's internal buffers
   (buffer identity is the slot), clobbering intermediates at replay.
   `reserve_slots_of` advances `next_slot` past every non-negative buffer slot after
   each scheduling (scan bodies, top-level compiles, and disk-cache loads).

## Fix landed: external co-tangents (grad w.r.t. tensors the body closes over)

Gradients w.r.t. **external inputs** of a scan — differentiable tensors the body closes
over (e.g. RNN weights), as opposed to the carry and the scanned sequence — were zero
under jit. The backward body's private-tape capture tracked only the carry and `x`
slots, so the pullback never computed the per-step contributions `dg_i` of an external
input, and the outer tape received nothing for it.

- **Discovery** (`stage_scan`): while the body is traced, a `tolk_of` hook
  (`st.scan_collectors`, a stack so nested scans propagate outward) records which
  *pre-existing* traced tensors the body reads — its free inputs. Registered per scan in
  `st.scan_closed`, keyed by `Obj.repr step` like `scan_stacks`. Constants the body
  captures are first touched inside the trace, so they never enter the list (they cannot
  be tracked by an enclosing grad anyway).
- **Accumulation** (`stage_scan_bwd`): the external inputs are tracked on the private
  tape, so the captured pullback emits `dg_i`; the backward loop totals each in a
  zero-seeded double-buffered accumulator (same parity scheme as the carry cotangents),
  the body writing `acc + dg_i` each iteration.
- **Delivery**: `E_scan_bwd` returns `scan_bwd_res` carrying `(tensor, cotangent)` pairs
  (`Rune_scan.Closed_ctan`); the reverse tape entry accumulates each into the outer tape
  for tensors it tracks (loop slots and constants never are, so spurious entries
  collected in nested scans are filtered out there).
- **Flat body stores**: the accumulation stores tripped a latent rangeify gap — a
  multi-d value stored into a loop's flat 1-D buffer leaves a rank-mismatched
  `INDEX(RESHAPE(buffer, …), flat_range)` that lowering cannot handle ("memory
  coalescing should be on INDEX, not INDEX"); the same pattern breaks multi-d carries
  and per-step outputs of the *forward* loop too. All body stores now flatten the value
  first (`store_flat`), so flat buffers meet flat values.

Verified by three new `test_jit.ml` cases: grad w.r.t. an external scalar weight and
external matrices (replay included), and a matrix (2-D) carry/output scan.

## Fix landed: CUDA misaligned address (error 716) on strided loop slots

On strict-alignment devices (CUDA, Metal) a vectorized kernel access through a pointer
whose address is not aligned to the access width faults with error 716. The loop
executor's per-iteration slot views pass `base + i * stride * itemsize` byte-offset
pointers to the body kernels, but the body kernels are compiled once assuming an
*aligned* base pointer — the offset is invisible to the compiler, and its vectorization
decision checks only index-expression divisibility (`coalesce_divides`), not the
runtime address.  When `stride * itemsize` is not a multiple of the widest vector
access (16 bytes, i.e. float4/half8) the offset view is misaligned for a vectorized
load or store.

`exec_loop` now routes such slots through an aligned scratch buffer allocated per slot:
the slice is copied in before the body runs (in-slots) and copied back after it (out-
slots).  memcpys have no alignment requirement and are stream-ordered with the kernel
launches (both use the NULL stream on CUDA).  Slots with stride ⋅ itemsize mod 16 = 0
use direct views as before — the common aligned case (e.g. carry rows of 4 floats =
16 bytes) keeps its vectorized accesses and has no overhead.

Scratch buffers are allocated once per loop call through the device allocator (whose
Lru_allocator recycles by size), and are reclaimed by GC finalisers like transient
views.

## Tests

`packages/rune/test/test_jit.ml` (the old `scan unrolls into the trace` is gone — the
scan no longer unrolls): 54 tests including forward scan matches eager (+replay); grad
of a tanh recurrence (carry+ys in the loss, fresh-data replay, `n=1`); grad with only
the stacked outputs or only the final carry in the loss (zero cotangents); multi-leaf
(record) carry; nested scan (forward and grad); captured weight in the body; vector
carry; external (closed-over) scalar weight and matrices; matrix carry.

## Known v1 restrictions (documented in 05-staged-scan.md)

pmap, symbolic body vars, and non-shape-stable carry are unsupported (explicit errors);
jvp/vmap of scan stay eager (correct, unrolled). Loop-containing programs never reach
`Jit_cache`'s disk cache: `store`'s `leaks_local_slot` check rejects them (their internal
buffers are not call arguments), and `load` reserves their slots on import.

## Debug helpers

- `RUNE_JIT_DEBUG=1`: cache/replay/transfer logs (small). `DEBUG=2`: tag-level
  `exec_loop` slot dump. Do **not** add `U.pp` dumps of calls/linears: `U.pp` unfolds
  shared subgraphs exponentially, which deadlocked `test_jit_cache` (its children print
  >64KB on stderr; `run_child` drains stdout before stderr, so the child blocks on a
  full pipe — latent in the test, consider draining concurrently).
