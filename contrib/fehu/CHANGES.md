# Changelog

All notable changes to Fehu are documented in this file. Releases up to
1.0.0~alpha3 are recorded in the repository's root `CHANGES.md`.

## Unreleased

- Fehu is a contrib package: it has its own version and changelog, and
  `opam install raven` no longer installs it.
- `Space.Box.sample` and `Space.Multi_discrete.sample` draw every dimension
  in one tensor operation instead of an OCaml loop reading each draw back to
  the host. The fallback for an unbounded `Box` dimension is unchanged.
  Samples for a given seed differ from before.
