# Changelog

All notable changes to Sowilo are documented in this file. Releases up to
1.0.0~alpha3 are recorded in the repository's root `CHANGES.md`.

## Unreleased

- Sowilo is a contrib package: it has its own version and changelog, and
  `opam install raven` no longer installs it.
- `resize`, `canny`, `threshold`, `invert`, and the HSV conversions combine
  with scalars instead of materializing full-size constant tensors.
