# Changelog

All notable changes to Sowilo are documented in this file. Releases up to
1.0.0~alpha3 are recorded in the repository's root `CHANGES.md`.

## Unreleased

- **Breaking:** `threshold` and `canny` return `'d Nx.bool_t` masks, and
  `structuring_element` and the morphological operations' `~kernel` are
  host booleans whose `true` entries select the neighbourhood. They returned
  float32 `0.`/`1.` and took a `uint8` kernel whose nonzero entries counted.
  Morphology takes boolean images as binary erosion and dilation, and
  `morphological_gradient` of a mask is the dilation without the erosion.
- `opening` and `closing` dilate, and erode, by the element reflected about
  its centre, so the opening is at most the image and both are idempotent
  for any element. For an element not symmetric about its centre they used
  the element itself, and `opening` of the row `[. X X]` by `[. X X]` was
  `[X X X]`. A `~kernel` not of rank 2 raises `Invalid_argument`.
- `dilate`, and `opening`, `closing` and `morphological_gradient` through it,
  count pixels outside the image as the dtype's least value. They counted them
  as zero, so a border pixel of an image with negative values, as
  `normalize`'s or `sobel`'s output, dilated to `0`.
- **Breaking:** images carry their device set, `('a, 'b, 'd) Nx.t` and
  `'d Nx.float32_t`. Each operation computes where its image lies and makes
  its kernels and constants beside it, so it runs on any set and its result
  lies on the image's. `structuring_element` gives a host value, and the
  morphological operations take a host `~kernel`, read once per call.
- `gaussian_blur` with a `sigma` of 0 or less takes a fixed kernel for a
  `ksize` up to 9 (`[1 2 1] / 4` for 3) and the Gaussian of sigma
  `0.3 * ((ksize - 1) / 2 - 1) + 0.8` beyond. It raises `Invalid_argument`
  for a NaN `sigma` or without a `ksize`; `unsharp_mask` and `canny` raise
  for such a `sigma`. It built its kernel from `0 / 0` and returned NaN.
- `sobel` and `laplacian` use their `?ksize`, 1, 3, 5 or 7 for `sobel` and
  any positive odd size for `laplacian`, and raise `Invalid_argument` for any
  other. Both ignored it. `laplacian`'s default is `ksize` 1, the
  4-neighbour aperture it always used; `~ksize:3` gives the 3x3 aperture
  `[2 0 2; 0 -8 0; 2 0 2]`.
- `to_grayscale` takes any image of 3 or more channels and ignores the
  channels after the third, such as alpha. It raised for 4 or more. It raises
  `Invalid_argument` for fewer channels or a rank below 3.
- `gaussian_blur`'s default `ksize` is `2 * ceil(3 * sigma) + 1`, as
  documented; it rounded. A `ksize` of 0 or less raises `Invalid_argument`,
  as documented; it took the default. A `sigma` whose default `ksize` exceeds
  `max_int` raises `Invalid_argument`; the size wrapped around.
- `box_blur` and `filter2d` with an even kernel size take each pixel's window
  from `k / 2` pixels before it to `k / 2 - 1` after, along each axis: the
  window started one pixel later.
- Sowilo is a contrib package: it has its own version and changelog, and
  `opam install raven` no longer installs it.
- `resize`, `canny`, `threshold`, `invert`, and the HSV conversions combine
  with scalars instead of materializing full-size constant tensors.
