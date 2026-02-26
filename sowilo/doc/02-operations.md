# Operations Reference

Every operation in sowilo, organized by category. All functions operate on
Rune float32 tensors with values in [0, 1] unless otherwise noted.

## Type Conversion and Preprocessing

### to_float

Converts a tensor to float32 and scales to [0, 1] by dividing by 255.

<!-- $MDX skip -->
```ocaml
let img = Nx_io.load_image "photo.png" |> Rune.of_nx |> Sowilo.to_float
(* uint8 [0, 255] -> float32 [0.0, 1.0] *)
```

### to_uint8

Scales from [0, 1] to [0, 255] and casts to uint8. Values are clipped to
[0, 1] before scaling.

<!-- $MDX skip -->
```ocaml
let result = Sowilo.to_uint8 processed_img
(* float32 [0.0, 1.0] -> uint8 [0, 255] *)
```

### normalize

Per-channel normalization: `(img - mean) / std`. The `mean` and `std` lists
must match the channel dimension length.

<!-- $MDX skip -->
```ocaml
(* ImageNet normalization *)
let normalized =
  Sowilo.normalize
    ~mean:[0.485; 0.456; 0.406]
    ~std:[0.229; 0.224; 0.225]
    img
```

Raises `Invalid_argument` if `mean` or `std` length does not match the
number of channels.

### threshold

Binary thresholding: returns 1.0 where the image exceeds the threshold, 0.0
elsewhere.

<!-- $MDX skip -->
```ocaml
(* Pixels > 0.5 become 1.0, rest become 0.0 *)
let binary = Sowilo.threshold 0.5 gray_img
```

## Color Space Conversion and Adjustment

### to_grayscale

Converts RGB to single-channel grayscale using ITU-R BT.601 weights:
`0.299 * R + 0.587 * G + 0.114 * B`. Input must have C >= 3. Output has
C = 1.

<!-- $MDX skip -->
```ocaml
let gray = Sowilo.to_grayscale rgb_img
```

### rgb_to_hsv / hsv_to_rgb

Convert between RGB and HSV color spaces. H is in [0, 1] (normalized from
[0, 360]), S and V are in [0, 1].

<!-- $MDX skip -->
```ocaml
let hsv = Sowilo.rgb_to_hsv img
(* ... manipulate hue, saturation, value channels ... *)
let rgb = Sowilo.hsv_to_rgb hsv
```

### adjust_brightness

Scales pixel values by a factor and clips to [0, 1].

<!-- $MDX skip -->
```ocaml
let brighter = Sowilo.adjust_brightness 1.3 img   (* 30% brighter *)
let darker = Sowilo.adjust_brightness 0.7 img     (* 30% darker *)
```

### adjust_contrast

Adjusts contrast around the per-channel mean. Factor 0 produces solid gray,
1 is the original image.

<!-- $MDX skip -->
```ocaml
let high_contrast = Sowilo.adjust_contrast 1.5 img
let low_contrast = Sowilo.adjust_contrast 0.5 img
```

### adjust_saturation

Adjusts color saturation via HSV. Factor 0 produces grayscale, 1 is the
original image.

<!-- $MDX skip -->
```ocaml
let vivid = Sowilo.adjust_saturation 1.5 img
let muted = Sowilo.adjust_saturation 0.5 img
```

### adjust_hue

Rotates hue by a delta in [-0.5, 0.5], corresponding to a full rotation of
the hue circle.

<!-- $MDX skip -->
```ocaml
let warm = Sowilo.adjust_hue 0.05 img
let cool = Sowilo.adjust_hue (-0.05) img
```

### adjust_gamma

Applies gamma correction: `img ** gamma`. Values less than 1.0 brighten,
greater than 1.0 darken.

<!-- $MDX skip -->
```ocaml
let brightened = Sowilo.adjust_gamma 0.5 img
let darkened = Sowilo.adjust_gamma 2.0 img
```

### invert

Inverts the image: `1.0 - img`.

<!-- $MDX skip -->
```ocaml
let negative = Sowilo.invert img
```

## Geometric Transforms

### resize

Resizes to target dimensions. Defaults to bilinear interpolation. Casts to
float32 internally for bilinear mode.

<!-- $MDX skip -->
```ocaml
let small = Sowilo.resize ~height:224 ~width:224 img
let nearest = Sowilo.resize ~interpolation:Nearest ~height:64 ~width:64 img
```

Raises `Invalid_argument` if height or width is not positive.

### crop

Extracts a rectangular region starting at (y, x) with the given dimensions.

<!-- $MDX skip -->
```ocaml
let region = Sowilo.crop ~y:50 ~x:100 ~height:200 ~width:300 img
```

Raises `Invalid_argument` if the region exceeds image bounds.

### center_crop

Crops a centered rectangle of the given size.

<!-- $MDX skip -->
```ocaml
let centered = Sowilo.center_crop ~height:200 ~width:200 img
```

Raises `Invalid_argument` if the crop size exceeds image dimensions.

### hflip / vflip

Flip horizontally (left to right) or vertically (top to bottom).

<!-- $MDX skip -->
```ocaml
let mirrored = Sowilo.hflip img
let upside_down = Sowilo.vflip img
```

### rotate90

Rotates by k * 90 degrees counter-clockwise. k defaults to 1. Negative
values rotate clockwise.

<!-- $MDX skip -->
```ocaml
let rotated_90 = Sowilo.rotate90 img              (* 90 CCW *)
let rotated_180 = Sowilo.rotate90 ~k:2 img         (* 180 *)
let rotated_cw = Sowilo.rotate90 ~k:(-1) img       (* 90 CW *)
```

### pad

Zero-pads the spatial dimensions. The tuple specifies (top, bottom, left,
right) padding. An optional `~value` parameter sets the fill value (defaults
to 0.0).

<!-- $MDX skip -->
```ocaml
let padded = Sowilo.pad (10, 10, 20, 20) img
let white_padded = Sowilo.pad ~value:1.0 (5, 5, 5, 5) img
```

## Spatial Filtering

### gaussian_blur

Isotropic Gaussian blur using separable convolution. `sigma` is required.
`ksize` defaults to `2 * ceil(3 * sigma) + 1`, capturing 99.7% of the
distribution.

<!-- $MDX skip -->
```ocaml
let blurred = Sowilo.gaussian_blur ~sigma:1.0 img
let blurred_5x5 = Sowilo.gaussian_blur ~sigma:1.5 ~ksize:5 img
```

Raises `Invalid_argument` if `ksize` is even or not positive.

### box_blur

Applies a ksize x ksize averaging filter.

<!-- $MDX skip -->
```ocaml
let averaged = Sowilo.box_blur ~ksize:3 img
let smooth = Sowilo.box_blur ~ksize:7 img
```

Raises `Invalid_argument` if `ksize` is not positive.

### median_blur

Applies a median filter. **Not differentiable**: uses sort internally,
gradient is zero almost everywhere.

<!-- $MDX skip -->
```ocaml
let denoised = Sowilo.median_blur ~ksize:3 img
```

Raises `Invalid_argument` if `ksize` is not a positive odd integer.

### filter2d

Applies a custom 2D convolution kernel of shape `[kH; kW]`. Applied
independently to each channel with Same padding.

<!-- $MDX skip -->
```ocaml
(* Sharpen kernel *)
let kernel = Rune.create Rune.float32 [| 3; 3 |]
  [| 0.; -1.; 0.; -1.; 5.; -1.; 0.; -1.; 0. |]
let sharpened = Sowilo.filter2d kernel img
```

### unsharp_mask

Sharpens by subtracting a Gaussian blur:
`img + amount * (img - gaussian_blur ~sigma img)`. `amount` defaults to 1.0.

<!-- $MDX skip -->
```ocaml
let sharp = Sowilo.unsharp_mask ~sigma:1.0 img
let very_sharp = Sowilo.unsharp_mask ~sigma:1.0 ~amount:2.0 img
```

## Morphological Operations

### structuring_element

Creates a structuring element of the given shape and size. The size is a
pair of positive odd integers (height, width).

Three shapes are available:
- `Rect` -- full rectangle
- `Cross` -- cross-shaped element
- `Ellipse` -- elliptical element

<!-- $MDX skip -->
```ocaml
let rect = Sowilo.structuring_element Rect (5, 5)
let cross = Sowilo.structuring_element Cross (3, 3)
let ellipse = Sowilo.structuring_element Ellipse (7, 7)
```

Raises `Invalid_argument` if height or width is not positive or not odd.

### erode / dilate

Erosion replaces each pixel with the minimum over the kernel-shaped
neighborhood. Dilation replaces with the maximum.

<!-- $MDX skip -->
```ocaml
let kernel = Sowilo.structuring_element Rect (5, 5) in
let eroded = Sowilo.erode ~kernel img
let dilated = Sowilo.dilate ~kernel img
```

### opening / closing

Opening (erode then dilate) removes small bright regions. Closing (dilate
then erode) fills small dark regions.

<!-- $MDX skip -->
```ocaml
let kernel = Sowilo.structuring_element Rect (5, 5) in
let opened = Sowilo.opening ~kernel binary_img
let closed = Sowilo.closing ~kernel binary_img
```

### morphological_gradient

The difference between dilation and erosion: `dilate - erode`. Highlights
edges.

<!-- $MDX skip -->
```ocaml
let kernel = Sowilo.structuring_element Rect (3, 3) in
let edges = Sowilo.morphological_gradient ~kernel img
```

## Edge Detection

All edge detection operations require grayscale input (C = 1).

### sobel

Computes Sobel gradients. Returns a `(gx, gy)` tuple where `gx` is the
horizontal gradient and `gy` is the vertical gradient. `ksize` defaults
to 3.

<!-- $MDX skip -->
```ocaml
let gx, gy = Sowilo.sobel gray in
let gx5, gy5 = Sowilo.sobel ~ksize:5 gray in

(* Compute gradient magnitude *)
let magnitude =
  Rune.sqrt (Rune.add (Rune.mul gx gx) (Rune.mul gy gy))
```

### scharr

Computes Scharr gradients, which are more rotationally accurate than Sobel.
Returns a `(gx, gy)` tuple.

<!-- $MDX skip -->
```ocaml
let gx, gy = Sowilo.scharr gray
```

### laplacian

Computes the Laplacian (sum of second spatial derivatives). `ksize` defaults
to 3.

<!-- $MDX skip -->
```ocaml
let lap = Sowilo.laplacian gray
let lap5 = Sowilo.laplacian ~ksize:5 gray
```

### canny

Canny edge detector. Returns 1.0 for edge pixels, 0.0 otherwise. `low` and
`high` are hysteresis thresholds (in [0, 1] since images are float32 in
[0, 1]). `sigma` controls the initial Gaussian blur and defaults to 1.4.

**Not differentiable**: uses non-maximum suppression and hysteresis
thresholding.

<!-- $MDX skip -->
```ocaml
let edges = Sowilo.canny ~low:0.2 ~high:0.6 gray
let tight = Sowilo.canny ~low:0.3 ~high:0.7 ~sigma:1.0 gray
```

## Differentiability Summary

Most operations are differentiable because they are built from standard
Rune tensor operations. The two exceptions are:

| Operation | Differentiable | Reason |
|-----------|---------------|--------|
| `median_blur` | No | Uses sort; gradient is zero almost everywhere |
| `canny` | No | Uses non-maximum suppression and hysteresis thresholding |

All other operations (filters, color transforms, geometric transforms,
morphology, threshold, sobel, scharr, laplacian) support `Rune.grad`.
