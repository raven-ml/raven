# Staged scan under jit — progress

Branch: `scan-jit`. Design notes: `packages/rune/doc/05-staged-scan.md`.

Goal: `Rune.jit (fun p -> ... Rune.scan ...)` (incl. `grad`) compiles the fold step once as a
loop in the compiled program instead of unrolling the whole trace.

## What is done (compiling, 43/44 rune tests pass)

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
  through it (debug prints under `DEBUG>=2`).
- `lib/schedule/rangeify.ml`: `split_store` keeps a precompiled `CALL` as a store value
  whole — it replaces the STORE as the AFTER's kernel dep instead of building a kernel
  around it.

## Known failures / next steps

1. **`test_scan_unrolls_inside_jit` still fails at replay** with
   `resolve: unbound PARAM` in `exec_loop`: the loop call's argument list in the final
   linear is `PARAM(slot 6..11)` (pm_replace_buf's numbering) instead of the substituted
   buffer nodes. `Schedule.resolve_linear_call_rule` only substitutes PARAMs when the top
   call's body is directly `LINEAR` (it is `FUNCTION(TUPLE(LINEAR))`, so it is skipped —
   the regular kernels' args never needed substitution because their args are buffers,
   but the loop call's args go through `call_arg_buffer_node` on the realize-arg AFTER
   and come out as PARAMs). Fix plan: substitute PARAM args against the top call's
   argument list (`ctx.replacements`) in `rune/lib/jit.ml` `trace_compile` right after
   the linear is captured — the slots index `U.as_call call`'s args consistently. The
   body's inner linear looks fine in the dump (kernels compiled, payload intact).
2. After the replay fix, verify the forward scan results, then the **backward pass**
   (`Rune.grad` of a scan under jit — `stage_scan_bwd` / reverse's `E_scan_bwd` thunk)
   and add tests (replace/upgrade the "scan unrolls" test; add grad-through-scan under
   jit, multi-leaf carry, nested scan, repeated calls, n=1).
3. **Hang after the rune test suite** (`dune runtest packages/rune` prints all results
   then hangs): investigate — likely the CPU executor's Domain pool not joining after
   the exception in `exec_loop`, or a later test binary (jit_cache/pmap) waiting. The
   failing test binary itself exits fine on its own.
4. Known v1 restrictions (documented in 05-staged-scan.md): pmap, symbolic body vars,
   non-shape-stable carry, and `Jit_cache` for loop programs are unsupported; jvp/vmap
   of scan stay eager (correct, unrolled).

## Debug helpers in place

- `RUNE_JIT_DEBUG=1`: dumps the compiled linear; `DEBUG=2`: `exec_loop` arg dump.
- Both guarded prints can stay or be removed once the replay fix lands.
