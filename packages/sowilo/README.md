# Sowilo

Differentiable computer vision for OCaml, built on [Rune](../rune/)

Sowilo provides image processing operations expressed purely through Rune
tensor operations. All operations are compatible with `Rune.grad`,
`Rune.jit`, and `Rune.vmap`.

## Quick Start

Load an image, detect edges, and save the result:

```ocaml
open Sowilo

let () =
  let img = Nx_io.load_image "photo.png" |> to_float in
  let gray = to_grayscale img in
  let edges = canny ~low:0.2 ~high:0.6 gray in
  Nx_io.save_image (to_uint8 edges) "edges.png"
```

## Features

- **Type conversion**: `to_float`, `to_uint8`, `normalize`, `threshold`
- **Color**: `to_grayscale`, `rgb_to_hsv`/`hsv_to_rgb`, `adjust_brightness`, `adjust_contrast`, `adjust_saturation`, `adjust_hue`, `adjust_gamma`, `invert`
- **Geometric transforms**: `resize` (nearest, bilinear), `crop`, `center_crop`, `hflip`, `vflip`, `rotate90`, `pad`
- **Spatial filters**: `gaussian_blur`, `box_blur`, `median_blur`, `filter2d`, `unsharp_mask`
- **Morphology**: `structuring_element` (Rect, Cross, Ellipse), `erode`, `dilate`, `opening`, `closing`, `morphological_gradient`
- **Edge detection**: `sobel` (returns gx, gy), `scharr`, `laplacian`, `canny`
- **Differentiable**: most operations support `Rune.grad` (exceptions: `canny`, `median_blur`)
- **Batch ready**: all operations handle `[H; W; C]` and `[N; H; W; C]` tensors

## Image Conventions

Images are Rune tensors with channels-last layout. Operations expect float32
values in [0, 1].

- Single image: `[H; W; C]` (height, width, channels)
- Batch: `[N; H; W; C]` (batch, height, width, channels)
- Grayscale: C = 1, RGB: C = 3, RGBA: C = 4

## Examples

- **01-grayscale** -- RGB to grayscale conversion
- **02-gaussian-blur** -- Gaussian blur with configurable sigma and kernel size
- **03-median-blur** -- Median filtering for noise removal
- **04-threshold** -- Binary thresholding
- **05-sobel** -- Sobel gradient computation (horizontal and vertical)
- **06-canny** -- Canny edge detection with hysteresis thresholds
- **07-morphology** -- Erosion, dilation with structuring elements

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
