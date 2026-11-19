# Changelog

All notable changes to Mynd are documented in this file.

## Unreleased

- Add colour models: `luma` (BT.601, on encoded values) and `luminance`
  (CIE Y of linear sRGB), exact on greys; `rgb_to_hsv` and `hsv_to_rgb`,
  Smith's hexcone with hue in turns, whose round trip has the identity as
  derivative at tied channels too; `linear_to_lab` and `lab_to_linear`,
  CIE L*a*b* relative to D65 with CSS Color 4's rational matrices.
- New library `mynd`, images over nx: `convert` between dtypes by Vulkan's
  normalized fixed-point rules, `srgb_to_linear` and `linear_to_srgb`,
  `premultiply` and `unpremultiply`, `sample` and `nearest` at positions
  under box, triangle, Keys' cubic and Lanczos kernels with an edge rule,
  `Transform`, band-limited `resize` and its `resize_weights`, `Box`, `Mask`
  and COCO's `Rle`, `components` and `nms`. Each is a composition of nx
  operations that rune differentiates, batches and compiles.
