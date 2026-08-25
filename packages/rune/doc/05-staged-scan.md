# Staged scan — smart jit compilation of `Rune.scan`

> Design notes for compiling `scan` (and its reverse pass) as a loop in the
> compiled program instead of an unrolled trace. Status: **implemented** —
> forward and reverse scans stage as `CALL(CUSTOM_FUNCTION "loop")` nodes.

## Today: eager scan, unrolled traces

`Rune.scan` is an ordinary OCaml loop over slices of `xs` (`lib/rune.ml`). It
performs no effect, so the jit tracer — which records one graph node per
intercepted Nx operation — sees every iteration. Under

<!-- $MDX skip -->
```ocaml
Rune.jit (fun p -> Rune.value_and_grad (module P) (loss_with_scan) p)
```

the trace contains `n` unrolled forward steps *and* `n` unrolled pullback
steps, because the reverse handler's tape thunks run inside the jit trace too.
The compiled program is `O(n · |body|)` in size and compile time; this is the
behaviour pinned by `test/test_jit.ml` (`scan unrolls into the trace`).

The goal is JAX `lax.scan`-style staging: trace the fold step **once**, compile
it once, and make the compiled program execute the loop — for the forward pass
and for the transposed (backward) pass — while everything outside jit keeps its
current eager semantics.

## Why this is tractable

Three architectural facts, established by reading the code, make this a
localized change rather than a redesign:

1. **The compiled program is a call list interpreted at runtime.** Tolk lowers
   the graph to a linear sequence of `CALL` nodes (`Slice` / `Copy` /
   `Program` kernels), dispatched one by one by `Tolk.Realize.run_linear`
   (`tolk/lib/engine/realize.ml`). Kernels are compiled independently per
   call. A loop can therefore be a **hierarchical call-list construct** — the
   body is its own linearized sub-list run `n` times with per-iteration buffer
   rebinding — without touching kernel codegen or renderers at all.

2. **Tracing is effect-based, and OCaml 5.2 made `Effect.Unhandled`
   catchable** (repo is on 5.5.0). `scan` can attempt a staged effect and fall
   back to the eager loop when no tracer is present — zero behaviour change for
   every existing code path.

3. **The reverse pass needs a scan-transpose rule** (the classic
   `lax.scan` construction), and rune already has every ingredient: `no_grad`
   taping suppression (`lib/gate.ml`), nestable handlers (nested grads
   already compose), private tapes, and a placeholder-allocating tracer.

## Design

### 1. A staged `E_scan` effect with eager fallback

`scan` performs a new rune-defined effect and falls back to the eager loop
when nothing claims it (`Effect.Unhandled`) or the claimer declines
(`Not_staged`, see below). The effects live entirely in rune — Nx's effect
type is extensible (`Nx_effect`), so **Nx needs no changes**. `E_scan`
carries an existentially packed carry and a type-erased body re-runner (the
same `Packed` GADT pattern jit already uses).

Three effects make up the protocol (`lib/rune_scan.ml`):

- **`E_scan`** — the scan itself. The jit stages it; each transformation
  handler (reverse, vmap, jvp, a pmap trace) claims it and runs the eager
  fold under a nested copy of itself, so the steps get taped/batched exactly
  as the unrolled loop always was. vmap's claim is unconditional by
  necessity: the body may close over batched tensors that appear in neither
  the carry nor the scanned input, so a scan can never be handed through
  vmap to a stager above it.
- **`E_scan_probe`** — "will the nearest `E_scan` claimer stage?". Every
  handler with an `E_scan` case must answer it: the jit answers `true` on a
  single-device trace and `false` under pmap; every transformation handler
  answers `false`. Reverse-mode asks before re-performing `E_scan`, because
  the staged-transpose tape entry it would record performs an `E_scan_bwd`
  that only a staging jit can answer — on `false` it folds eagerly instead.
  Unhandled means `false`.
- **`Not_staged`** — a stager's decline. Shape instability is only visible
  after tracing the body, so a jit that claimed `E_scan` may still
  discontinue it with `Not_staged`; every performer of `E_scan` treats it
  like `Effect.Unhandled` and folds eagerly. The probe is an optimistic
  answer, not a promise.

The resulting composition matrix, with a scan in `f`:

| Composition | Behaviour |
| --- | --- |
| `jit f`, `jit (grad f)` | staged loop (reversed loop for the transpose) |
| `grad f`, `vmap f`, `jvp f` (no jit) | eager fold, per-step taping/batching |
| `jit (vmap f)`, `jit (vmap (grad f))` | unrolls into the trace |
| `pmap f`, `pmap (grad f)` | unrolls into each shard's trace |
| `grad (grad f)`, `hvp` | eager fold, per-step taping (no staged transpose-of-transpose) |
| shape-unstable carry under `jit` | declines, unrolls into the trace |

### 2. Forward loop staging in `jit.ml`

The jit handler answers `E_scan` by tracing the body **once** with fresh
placeholder (carry, x) tensors, then emitting a Tolk `Loop` node:

- **Body capture.** The tracer's table already maps placeholder → graph node;
  a table checkpoint before/after the body run extracts the body subgraph
  (its free inputs are the body's placeholder tensors, which become the loop's
  input slots).
- **Static trip count.** `n` is the static shape of `xs` along axis 0. `scan`
  needs no runtime exit condition — this is what distinguishes it from
  `cond`/`while_loop`, which require genuinely data-dependent control flow
  (see *Out of scope*).
- **Loop outputs.** The final carry and the stacked `ys` come back as fresh
  placeholders, so the surrounding trace proceeds exactly as if the ops had
  been recorded inline.
- **Carried buffers.** The loop declares its carried buffers (the carry, and a
  **carry stack** written at slot `i` each iteration — see §3).

Nested `scan`s are handled by the same case recursively.

### 3. Reverse-mode: the scan transpose (`reverse.ml`)

The reverse handler records **one tape entry** for a staged scan instead of
`n`. When that entry runs at backward time (still inside the jit trace), it
performs `E_scan_bwd`, which jit lowers to a second `Loop` node running the
body in reverse. The pullback body is captured with existing machinery:

1. Run the body once with the loop's slot tensors as arguments under
   `no_grad` (outer taping suppressed) and a **private reverse tape**. The
   forward ops land flat in the outer trace — this interval becomes the
   backward body's *residual recomputation* (we hold `carry_i` and `x_i`, so
   all step-`i` intermediates are recomputable from them).
2. Create placeholder cotangents (`dcarry_next`, `dy_i`) via traced creation
   effects; they are free inputs, i.e. loop slots of the backward loop.
3. Replay the private tape with those cotangents under `no_grad`; the ops
   performed land in the outer trace as the *pullback interval*.
4. `E_scan_bwd` carries both intervals plus slot metadata; jit reparents them
   into the backward `Loop` node and binds its slots to the buffers shared
   with the forward loop.

The forward loop must carry a **carry stack** (it appends `carry_i` at slot
`i`; the backward loop reads slot `n-1-j` at iteration `j`), because `carry_i`
is not recoverable from `carry_{i+1}` in general. This is O(n · |carry|)
memory — the same price JAX pays for `lax.scan` under `grad`.

**External inputs** (differentiable tensors the body closes over, e.g. RNN
weights) transpose differently from the carry: each step contributes a
per-step cotangent and the results *sum* over steps rather than threading
through them. The forward staging discovers them (a `tolk_of` hook records the
pre-existing traced tensors the body reads); the backward capture tracks them
on the private tape so the pullback emits their contributions, and the
backward loop totals each in a zero-seeded accumulator carried in a double
buffer. The totals travel back in the `E_scan_bwd` result and are accumulated
into the enclosing tape for the tensors it tracks.

Outside jit, none of this runs: the probe routes to the eager loop and the
ordinary per-step taping.

### 4. Forward-mode and vmap fold eagerly

`forward.ml` (jvp) and `vmap.ml` have no staged rule: each claims `E_scan`
and runs the eager fold under a nested copy of itself, so the steps inherit
tangent pairing and batching exactly as the unrolled loop did. The claims
cannot fall through to a stager above: a vmap'd body may read batched
closure captures the scan request never mentions, and a jvp'd body reads its
tangent pairing from the handler's state — in both cases the step ops must
flow through the transformation handler, which a staged body trace would
bypass. (Reverse is different because its pass is a second, reversed loop —
that is what stages, via the probe.)

### 5. The Tolk `Loop` construct

The Tolk work is the bulk of the effort but stays within one subsystem:

- **Graph.** A `Loop` node that is opaque to outer graph rewrites — its body
  subgraph is stored out of band, never flattened into the outer graph. v1
  does no optimization passes on bodies (kernels are compiled per call anyway,
  so the body subgraph gets its own linearize + codegen exactly like a
  standalone program).
- **Planner.** Carried and stack buffers allocated once, outside the loop.
  The carry is **double-buffered** to break the cyclic read-after-write
  dependency; the executor swaps bindings between iterations. Body temporaries
  are planned once and reused across iterations (execution is sequential, so
  liveness never spans an iteration boundary). The loop index is a runtime
  symbolic var (`var_vals`, already supported by the linearizer), so
  per-iteration slice offsets (`x_i`, stack slot `i`, `ys` row `i`) reach the
  kernels as values.
- **Linearizer.** Emits `LOOP_BEGIN` / `LOOP_END` markers around the body's
  call sub-list, with slot-rebinding info.
- **Executor.** `run_linear` interprets the markers: on CPU, rebind and
  dispatch the sub-list `n` times. On CUDA, relaunch the recorded sub-graph
  per iteration using the existing diff-patch address-update machinery
  (`dyn`/`updatable` in `realize.ml`); batching can still coalesce within an
  iteration, just not across the loop boundary.

### 6. Replay, caching, devices

- **Signatures.** The loop adds `(carry shapes, xs shape, n)` to the compiled
  signature; a new `n` retraces and recompiles (cheap — tracing is one body
  run, not `n`).
- **Persistent cache.** Loop-containing programs are never persisted: the
  loop call's slot buffers keep process-local global slots, which
  `Jit_cache.store`'s normalization guard refuses to export. Serializing
  hierarchical linears can come later.
- **pmap.** A multi-device trace answers the staging probe with `false` and
  unrolls a directly performed scan into the trace; sharded staged loops are
  deferred.
- **Shape stability.** The staged path requires the body to return a carry of
  the shapes it received (the single prototype run stands for every
  iteration). Instability declines staging with `Not_staged` and the fold
  **unrolls** — graceful degradation to the pre-staging behaviour.

## Scope and effort

| Piece | Where | Size | Novelty |
| --- | --- | --- | --- |
| `E_scan` + probe + eager fallback | rune `lib/rune.ml` | small | plumbing |
| Forward loop staging + body capture | rune `lib/jit.ml` | ~300–500 lines | moderate |
| Scan transpose (carry stack, private-tape capture, `E_scan_bwd`) | rune `lib/reverse.ml` | ~300–400 lines | the subtle one |
| `Loop` graph node + planner + linearizer + executor | tolk (engine/schedule) | ~1500–2500 lines + tests | the bulk |
| Nx | — | **zero** | — |
| Kernel codegen / renderers | — | **zero** | — |

A focused multi-week project. The rune-side transpose follows the
well-trodden `lax.scan` construction; the Tolk-side planner (carried-buffer
lifetimes) and the CUDA per-iteration path are the riskiest parts.

## Out of scope

- **`cond` and `while_loop`**: these have data-dependent predicates and a
  runtime exit condition — genuinely harder than scan's static trip count.
  They would need real runtime branching, a different project.
- **In-shader loops** (loop constructs inside the generated kernel code,
  dynamic extents, full jaxpr-style IR): this *would* be a substantial Tolk
  redesign, and is unnecessary — the runtime already executes a call list, so
  the loop belongs there.
