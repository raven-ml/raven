# Changelog

All notable changes to Mynd are documented in this file.

## Unreleased

- New library `mynd`, images over nx: `convert` between dtypes by Vulkan's
  normalized fixed-point rules, `srgb_to_linear` and `linear_to_srgb`,
  `premultiply` and `unpremultiply`, `sample` and `nearest` at positions
  under box, triangle, Keys' cubic and Lanczos kernels with an edge rule,
  `Transform`, band-limited `resize` and its `resize_weights`, `Box`, `Mask`
  and COCO's `Rle`, `components` and `nms`. Each is a composition of nx
  operations that rune differentiates, batches and compiles.
