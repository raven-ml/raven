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

## Tests

`packages/rune/test/test_jit.ml` (the old `scan unrolls into the trace` is gone — the
scan no longer unrolls): forward scan matches eager (+replay); grad of a tanh recurrence
(carry+ys in the loss, fresh-data replay, `n=1`); grad with only the stacked outputs or
only the final carry in the loss (zero cotangents); multi-leaf (record) carry; nested
scan (forward and grad); captured weight in the body; vector carry.

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
