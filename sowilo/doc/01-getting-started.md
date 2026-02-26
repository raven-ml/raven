# Getting Started

This guide covers loading images, understanding image conventions, building
your first processing pipeline, and saving results.

## Installation

<!-- $MDX skip -->
```bash
opam install sowilo
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build sowilo
```

## Loading Images

Sowilo operates on Rune tensors. Load an image with `Nx_io`, convert it to
a Rune tensor, then to float32:

<!-- $MDX skip -->
```ocaml
open Sowilo

let img =
  Nx_io.load_image "photo.png"  (* Nx uint8 tensor [H; W; C] *)
  |> Rune.of_nx                 (* Rune uint8 tensor *)
  |> to_float                   (* Rune float32 tensor in [0, 1] *)
```

`to_float` divides by 255 and casts to float32. This is the standard input
format for all sowilo operations.

## Image Conventions

**Layout.** Images use channels-last layout:
- Single image: `[H; W; C]` (height, width, channels)
- Batch: `[N; H; W; C]` (batch, height, width, channels)

**Channel counts.** Grayscale has C = 1, RGB has C = 3, RGBA has C = 4.

**Value range.** Operations expect float32 values in [0, 1]. Use `to_float`
to convert from integer representations and `to_uint8` to convert back:

<!-- $MDX skip -->
```ocaml
(* uint8 [0, 255] -> float32 [0, 1] *)
let float_img = to_float uint8_img

(* float32 [0, 1] -> uint8 [0, 255] (clips to [0, 1] first) *)
let uint8_img = to_uint8 float_img
```

## Your First Pipeline

Load an image, convert to grayscale, blur, and detect edges:

<!-- $MDX skip -->
```ocaml
open Sowilo

let () =
  let img = Nx_io.load_image "photo.png" |> Rune.of_nx |> to_float in

  let edges =
    img
    |> to_grayscale                  (* RGB -> single channel *)
    |> gaussian_blur ~sigma:1.0      (* smooth noise *)
    |> canny ~low:0.2 ~high:0.6     (* detect edges *)
  in

  (* Save: convert back to uint8, then to Nx for I/O *)
  Nx_io.save_image (Rune.to_nx (to_uint8 edges)) "edges.png"
```

Operations compose naturally with `|>`. Each takes a tensor and returns a
tensor, so you can chain as many as you need.

## Color Adjustments

Adjust brightness, contrast, saturation, hue, and gamma:

<!-- $MDX skip -->
```ocaml
open Sowilo

let () =
  let img = Nx_io.load_image "photo.png" |> Rune.of_nx |> to_float in

  (* Each function takes a factor and an image *)
  let bright = adjust_brightness 1.3 img in
  let contrasted = adjust_contrast 1.5 img in
  let saturated = adjust_saturation 1.2 img in
  let warm = adjust_hue 0.05 img in
  let gamma = adjust_gamma 0.8 img in
  let negative = invert img in

  ignore (bright, contrasted, saturated, warm, gamma, negative)
```

## Geometric Transforms

Resize, crop, flip, rotate, and pad:

<!-- $MDX skip -->
```ocaml
open Sowilo

let () =
  let img = Nx_io.load_image "photo.png" |> Rune.of_nx |> to_float in

  let small = resize ~height:224 ~width:224 img in
  let cropped = crop ~y:50 ~x:100 ~height:200 ~width:200 img in
  let centered = center_crop ~height:200 ~width:200 img in
  let flipped = hflip img in
  let upside_down = vflip img in
  let rotated = rotate90 img in              (* 90 degrees counter-clockwise *)
  let rotated_cw = rotate90 ~k:(-1) img in   (* 90 degrees clockwise *)
  let padded = pad (10, 10, 10, 10) img in   (* top, bottom, left, right *)

  ignore (small, cropped, centered, flipped, upside_down, rotated, rotated_cw, padded)
```

`resize` defaults to bilinear interpolation. Pass `~interpolation:Nearest`
for nearest-neighbor.

## Morphological Operations

Build structuring elements and apply morphological operations:

<!-- $MDX skip -->
```ocaml
open Sowilo

let () =
  let img = Nx_io.load_image "photo.png" |> Rune.of_nx |> to_float in
  let gray = to_grayscale img in
  let binary = threshold 0.5 gray in

  (* Create a 5x5 rectangular structuring element *)
  let kernel = structuring_element Rect (5, 5) in

  let eroded = erode ~kernel binary in
  let dilated = dilate ~kernel binary in
  let opened = opening ~kernel binary in
  let closed = closing ~kernel binary in
  let grad = morphological_gradient ~kernel binary in

  ignore (eroded, dilated, opened, closed, grad)
```

Three kernel shapes are available: `Rect` (full rectangle), `Cross`
(cross-shaped), and `Ellipse` (elliptical). The size must be a pair of
positive odd integers.

## Edge Detection

Sowilo provides four edge detection methods:

<!-- $MDX skip -->
```ocaml
open Sowilo

let () =
  let img = Nx_io.load_image "photo.png" |> Rune.of_nx |> to_float in
  let gray = to_grayscale img in

  (* Sobel: returns horizontal and vertical gradients *)
  let gx, gy = sobel gray in

  (* Scharr: more rotationally accurate than Sobel *)
  let sx, sy = scharr gray in

  (* Laplacian: sum of second derivatives *)
  let lap = laplacian gray in

  (* Canny: binary edge map with hysteresis thresholding *)
  let edges = canny ~low:0.2 ~high:0.6 gray in

  ignore (gx, gy, sx, sy, lap, edges)
```

`sobel` and `scharr` return `(gx, gy)` tuples. All edge detectors require
grayscale input (C = 1).

## Saving Results

Convert back to uint8 and use `Nx_io.save_image`:

<!-- $MDX skip -->
```ocaml
let save result path =
  Nx_io.save_image (Rune.to_nx (to_uint8 result)) path
```

## Displaying with Hugin

Use Hugin for visualization:

<!-- $MDX skip -->
```ocaml
let () =
  let img = Nx_io.load_image "photo.png" |> Rune.of_nx |> to_float in
  let gray = to_grayscale img in
  let edges = canny ~low:0.2 ~high:0.6 gray in

  let fig = Hugin.figure ~width:1000 ~height:500 () in

  let ax1 = Hugin.subplot ~nrows:1 ~ncols:2 ~index:1 fig in
  ignore
    (ax1
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx img)
    |> Hugin.Axes.set_title "Original");

  let ax2 = Hugin.subplot ~nrows:1 ~ncols:2 ~index:2 fig in
  ignore
    (ax2
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx edges)
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Canny Edges");

  Hugin.show fig
```

## Next Steps

- [Operations Reference](02-operations/) -- every operation with detailed examples
- [Pipelines and Integration](03-pipelines/) -- composing pipelines, batch processing, deep learning
