# tolk

tolk is a port of tinygrad's compiler. `tolk` takes a sink and a target and
returns data: source, binaries, memory plans and schedules. `tolk.engine`
runs that data on rig.

## The reference

- `_tinygrad_next` at the repository root, at the commit `DIVERGENCES.md`
  pins. Never a released tinygrad package, and no internet sources unless
  asked.
- tinygrad decides structure: which module owns a pass, the pass boundaries
  and their order, where logic lives. Keep emitted source diffable against
  tinygrad's for covered kernels.
- Numerics follow nx's documented meaning, decided in rune's lowering with a
  failing rune test. tinygrad has bugs; a value it computes is a datapoint.
- Each difference is an entry in `DIVERGENCES.md`, admitted only for an
  OCaml constraint, a named raven consumer, or the executor's contract.
  Taste and unmeasured speed are not reasons.
- tolk stays at tinygrad's low-level uops: no matmul, contraction, attention
  or scan node.

## Changing it

- Prefer deletions, merges and ownership fixes to new code. A new IR, pass
  or layer needs proof that the current one cannot hold tinygrad's
  semantics.
- No special case that makes one kernel's source match. When source differs,
  look for the owner in this order: `opt/` or a late pass, the linearizer,
  the generic renderer.
- BEAM search is no answer to a performance problem.
- `.mli` and comments explain from first principles and never mention
  tinygrad. `DIVERGENCES.md` and `test/REGRESSIONS.md` do.

## Tests

- Goldens come from `uv run packages/tolk/test/gen/generate.py`, never by
  hand. A failing golden stays failing until the compiler is fixed.
- `test/REGRESSIONS.md` maps each tinygrad test to the test that keeps its
  behaviour, or to why it is dropped.
- A suite run by hand needs `CACHEDB=cache`. Without it, a suite walks the
  whole default disk cache and looks hung.
- A change to scheduling or the storage phase also runs the rune and jera
  suites.
