# Reviewing rune

## Design

- A rule written for one operation: would the general mechanism (a linear
  function's transpose, a jvp) cover it?
- A fix in the lowering: does the fact belong in nx's documented meaning or
  in tolk instead?
- A fix to a compiled call's cache key: does the key need the input, or
  should the state it keys on not exist?
- A library-side need for `jit`: programs compile, libraries don't.

## Semantics

References: the derivation, written out, or the paper that states it;
IEEE 754-2019 for the agreement classes the `.mli` states between eager
and compiled runs.

Failure here is a wrong derivative; a compiled result that differs from
eager beyond its class; a compiled call reused after something that changes
its program, or recompiled on every call; `vmap` lanes mixed; a random
stream that differs between eager and compiled.

- A new or changed derivative rule against finite differences, at real and
  complex dtypes, composed with `vmap` and `jit`.
- A compiled call's cache key: shapes, dtypes, placement, widths, tolk
  settings.
- State kept per domain where two domains can call.
- A lowering that departs from tinygrad: a `DIVERGENCES.md` entry, its
  reason, its pinning test.

## Skip

`test/golden/`, `bench/rune.thumper`.
