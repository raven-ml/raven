# Getting Started with sowilo

This guide shows you how to use sowilo for image processing and computer vision.

## Installation

Sowilo isn't released yet. When it is, you'll install it with:

```bash
opam install sowilo
```

For now, build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build sowilo
```

## Your First Image Processing Pipeline

Here's a working example that loads an image and detects edges:

```ocaml
open Sowilo

let () =
  (* Load image as nx array *)
  let img = Nx_io.load_image "photo.jpg" in
  
  (* Convert to Rune tensor for processing *)
  let tensor = Rune.of_bigarray (Nx.to_bigarray img) in
  
  (* Apply operations *)
  let processed = 
    tensor
    |> to_grayscale              (* Convert to grayscale *)
    |> gaussian_blur ~ksize:5 ~sigma:1.0  (* Smooth *)
    |> canny ~low:50. ~high:150.         (* Detect edges *)
  in
  
  (* Convert back to nx and save *)
  let result = Nx.of_bigarray (Rune.to_bigarray processed) in
  Nx_io.save_image result "edges.png"
```

## Key Concepts

**Tensors, not custom types.** Images are just tensors with shape `[height; width; channels]`. This means any tensor operation works on images.

**Dimension conventions:**
- Single image: `[H; W; C]` where C is 1 (grayscale) or 3 (RGB)
- Batch of images: `[N; H; W; C]` for batch size N

**Type safety.** Operations enforce correct tensor types:
- Most filters work on `float32` tensors in range [0, 1]
- Use `to_float` and `to_uint8` to convert between formats
- Some operations (like Sobel) output `int16` for derivatives

**Differentiable.** Since sowilo is built on Rune, all operations can be differentiated. Train neural networks that include classical CV operations.

## Common Operations

```ocaml
(* Color space conversions *)
let gray = to_grayscale color_img
let bgr = rgb_to_bgr rgb_img

(* Type conversions *)
let float_img = to_float uint8_img  (* [0,255] -> [0.0,1.0] *)
let uint_img = to_uint8 float_img   (* [0.0,1.0] -> [0,255] *)

(* Basic transformations *)
let flipped = flip_vertical img
let cropped = crop img ~x:100 ~y:50 ~width:200 ~height:150
let resized = resize img ~height:224 ~width:224 ~interpolation:Bilinear

(* Filtering *)
let blurred = gaussian_blur img ~ksize:5 ~sigma:1.0
let denoised = median_blur img ~ksize:3
let averaged = box_filter img ~ksize:3

(* Morphology *)
let kernel = get_structuring_element Rect ~ksize:3
let eroded = erode img kernel
let dilated = dilate img kernel

(* Thresholding *)
let binary = threshold img ~thresh:128. ~maxval:255. ~typ:Binary

(* Edge detection *)
let grad_x, grad_y = sobel img ~dx:1 ~dy:0 ~ksize:3
let edges = canny img ~low:100. ~high:200.
```

## Image I/O

Sowilo works with nx for image I/O:

```ocaml
(* Load images *)
let img = Nx_io.load_image "input.jpg"

(* Save images *) 
Nx_io.save_image processed "output.png"

(* Display with hugin *)
let fig = Hugin.imshow img ~title:"Processed Image" in
Hugin.show fig
```

## Working with Batches

Process multiple images at once:

```ocaml
(* Stack images into batch: [N; H; W; C] *)
let batch = Rune.Tensor.stack images ~axis:0

(* Apply operations - they broadcast over batch dimension *)
let processed_batch = gaussian_blur batch ~ksize:3 ~sigma:1.0

(* Split back to individual images *)
let processed_images = Rune.Tensor.unstack processed_batch ~axis:0
```

## Integration with Deep Learning

Sowilo operations can be part of neural networks:

```ocaml
(* Preprocessing for neural network *)
let preprocess img =
  img
  |> to_float
  |> resize ~height:224 ~width:224 ~interpolation:Bilinear
  |> to_grayscale
  |> gaussian_blur ~ksize:3 ~sigma:0.5

(* Use in training - gradients flow through! *)
let augmented = Model.forward model (preprocess input)
```

## Next Steps

Check out the examples in `sowilo/example/` for complete image processing pipelines including:
- Edge detection workflows
- Image filtering and denoising  
- Morphological operations
- Threshold techniques

When Rune's JIT compiler lands, these operations will automatically compile to efficient GPU kernels.