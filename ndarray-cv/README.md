# ndarray-cv

Computer vision extensions for Ndarray

`ndarray-cv` provides a suite of image processing and computer vision operations
built on top of Ndarray tensors.

## Features

- Basic image manipulations: flip_vertical, flip_horizontal, crop
- Color space conversions: to_grayscale, rgb_to_bgr, bgr_to_rgb
- Data type conversions: to_float, to_uint8
- Image resizing (nearest‑neighbor and bilinear)
- Filtering: gaussian_blur, median_blur
- Morphological operations: erode, dilate
- Edge detection: sobel, canny
- Thresholding: threshold
- Additional CV routines leveraging Ndarray

## Quick Start

```ocaml
open Ndarray
open Ndarray_io
open Ndarray_cv

(* Load an image into a uint8 tensor *)
let img = load_image "input.jpg"

(* Convert to grayscale *)
let gray = to_grayscale img

(* Apply Gaussian blur *)
let blur = gaussian_blur ~ksize:5 gray

(* Detect edges with Sobel filter *)
let edges = sobel ~dx:1 ~dy:0 ~ksize:3 blur

(* Save the processed image *)
save_image edges "edges.png"
```

## Contributing

See the [Raven monorepo README](../README.md) for guidelines on contributing.

## License

ISC License. See [LICENSE](../LICENSE) for details.
