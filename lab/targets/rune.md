# Target: rune

Tensor computation with autodiff and jit — the jax analog. rune is an nx
backend where each operation raises an effect; autodiff replays the forward
effects with gradient bookkeeping, and jit handles all effects to build and
compile a computation graph.

## Commands

- **Build:** `dune build packages/rune/bench/bench_rune.exe` — building the
  bench exe pulls in `packages/rune/lib` transitively.
- **Test (correctness gate):** `dune build @packages/rune/test/runtest` —
  scoped to the test dir; it does not run the benches.
- **Baseline:** `packages/rune/bench/rune.thumper` — the committed baseline
  the session refreshes at setup and ratchets on keeps.
- **BENCH:** the built executable, run **directly**, never through `dune exec`:
  `<WT>/_build/default/packages/rune/bench/bench_rune.exe`
  Suite name is `rune`; lab subset is `--tag lab`. Running the exe directly is
  required, not stylistic: `dune exec` sets `INSIDE_DUNE`, which makes
  `--bless --baseline PATH` write `PATH.corrected` instead of `PATH` and
  suppresses the corrected-file write the ratchet needs, and it forces a `--`
  separator that would swallow thumper's own flags.

## In scope (may edit)

- `packages/rune/lib/**` — chiefly `autodiff.ml` (reverse/forward mode, the
  tape), `nx_rune.ml` (the effect-raising backend and its C fallback), and the
  jit path.

## Read-only / never touch

- `packages/rune/bench/**`, `packages/rune/test/**`, every `*.thumper`.
- nx's `lib/` — rune calls into nx, but a rune session optimizes one layer at
  a time; nx is a separate target.
- Anything outside `packages/rune/lib/`.

## Perf context

- Two cost centers: per-operation effect-dispatch and tape overhead (visible in
  the `DeepChain/chain grad` case, where 100 sequential elementwise ops on a
  small tensor make per-op overhead dominate), and jitted execution (the `Jit`
  group — compilation is hoisted into bench setup and is not timed; the jit
  kernel cache lives in the returned closure and on disk under
  `$XDG_CACHE_HOME/tolk/rune_jit`).
- AD replays the forward effects; reducing allocation on the tape or in the
  replay is the usual win.
- Do not change gradient values — the tests pin them.
