# Target: kaun

Neural networks and training utilities on rune — the flax analog. This is the
one target that spans two packages, deliberately: the headline train step
spends most of its time in `Vega.adam_step`, and the optimizer lives in the
separate `vega` package, not `kaun/lib` — so vega is where the optimizer win
is.

## Commands

- **Build:** `dune build packages/kaun/bench/bench_kaun.exe` — building the
  bench exe pulls in `packages/kaun/lib` and `packages/vega` transitively.
- **Test (correctness gate):** `dune build @packages/kaun/test/runtest` —
  scoped to the test dir; it does not run the benches.
- **Baseline:** `packages/kaun/bench/kaun.thumper` — the committed baseline
  the session refreshes at setup and ratchets on keeps.
- **BENCH:** the built executable, run **directly**, never through `dune exec`:
  `<WT>/_build/default/packages/kaun/bench/bench_kaun.exe`
  Suite name is `kaun`; lab subset is `--tag lab`. Running the exe directly is
  required, not stylistic: `dune exec` sets `INSIDE_DUNE`, which makes
  `--bless --baseline PATH` write `PATH.corrected` instead of `PATH` and
  suppresses the corrected-file write the ratchet needs, and it forces a `--`
  separator that would swallow thumper's own flags.

## In scope (may edit)

- `packages/kaun/lib/**` — layers, ptree traversal, loss.
- `packages/vega/**` — the optimizer package (adam/sgd init and step).

## Read-only / never touch

- `packages/kaun/bench/**`, `packages/kaun/test/**`, any `packages/vega` tests
  or benches, every `*.thumper`.
- rune and nx `lib/` — a kaun session optimizes one layer at a time; those are
  separate targets.
- Anything outside `packages/kaun/lib/` and `packages/vega/`.

## Perf context

- The train step is `Rune.value_and_grad` (out of scope) + `Vega.adam_step`
  (in scope) + the bench's own ptree map (read-only). The in-scope wins are the
  vega optimizer update — per-parameter state allocation, buffer reuse — and
  kaun-side layer forward/backward.
- The isolated `Linear/linear fwd` and `Linear/linear fwd+bwd` cases and the
  `Conv/conv train step` case exercise more of `kaun/lib` (conv + pool layers)
  than the MLP headline does.
- Loss and layer outputs are pinned by tests — never change what they compute.
