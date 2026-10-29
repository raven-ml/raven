# Reviewing rune

## Design

- A rule written for one operation: would the general mechanism (a linear
  function's transpose, a jvp, a program's derivation) cover it?
- A construct handled outside programs: every construct traces its
  functions at the call, and its rules transform those programs.
- A fix in the lowering: does the fact belong in nx's stated meaning
  (`Nx_kernel`) or in tolk instead?
- A fix to a compiled call's key or variant: does the key need the input,
  or should the state it keys on not exist?
- A library-side need for `jit`: programs compile, libraries don't.

## Semantics

References: the derivation, written out, or the paper that states it;
IEEE 754-2019 for the agreement classes rune.mli's Numerics section states
between eager and compiled runs. JAX's documented semantics for a
transformation, tinygrad's decomposition for a lowering, are datapoints.

Failure here is a wrong derivative; a compiled result that differs from
eager beyond its class; a compiled call reused after something that changes
its program, or retraced on every call; `vmap` lanes mixed, or a held lane
that reaches another's value or derivative; a branch not taken that
computes; a total or a probe that counts a step twice; a random stream
that differs between eager and compiled.

- A new or changed derivative rule against finite differences, at real and
  complex dtypes, composed with `vmap` and `jit`.
- A construct's rule under each transformation, and nested in another
  construct's step.
- A compiled call's key (rune.mli's Keys) and variant (Compiled programs):
  each input that changes the program is in one of them.
- State kept per domain where two domains can call.
- A departure from JAX or tinygrad: a `DIVERGENCES.md` entry naming the
  function, its reason, its pinning test. An entry whose code is gone goes.

## Skip

`test/golden/`, `bench/*.thumper`, `bench/compile/*.thumper`.
