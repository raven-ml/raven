# Changelog

All notable changes to Fehu are documented in this file. Releases up to
1.0.0~alpha3 are recorded in the repository's root `CHANGES.md`.

## Unreleased

- **Breaking:** `Info.null`, `bool`, `int`, `float`, `string`, `int_array`,
  `float_array` and `bool_array` are removed: write the `Value` constructor,
  as `Value.Int 3`, with `Array.copy` where the array later changes.
- **Breaking:** `Gae.compute_from_values` is removed: pass `Gae.compute` a
  `next_values` built from `values` and the last value estimate.
- `Value.pp` writes a float as the shortest decimal that reads back, the same
  text on every system: `1000000` where it wrote `1e+06`, or `1e+006` under
  Windows, and every digit where it kept six.
- **Breaking:** spaces, observations and buffers hold host tensors, as
  `Nx.host Nx.float32_t`, and draw from `Nx.Rng`: samples for a seed differ
  from before.
- Fehu is a contrib package: it has its own version and changelog, and
  `opam install raven` no longer installs it.
- `Space.Box.sample` and `Space.Multi_discrete.sample` draw every dimension
  in one tensor operation instead of an OCaml loop reading each draw back to
  the host. The fallback for an unbounded `Box` dimension is unchanged.
  Samples for a given seed differ from before.
