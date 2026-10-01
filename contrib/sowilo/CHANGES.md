# Changelog

All notable changes to Sowilo are documented in this file. Releases up to
1.0.0~alpha3 are recorded in the repository's root `CHANGES.md`.

## Unreleased

- `box_blur` and `filter2d` with an even kernel size take each pixel's window
  from `k / 2` pixels before it to `k / 2 - 1` after, along each axis: the
  window started one pixel later.
- Sowilo is a contrib package: it has its own version and changelog, and
  `opam install raven` no longer installs it.
- `resize`, `canny`, `threshold`, `invert`, and the HSV conversions combine
  with scalars instead of materializing full-size constant tensors.
