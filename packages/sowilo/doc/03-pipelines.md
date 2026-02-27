# Pipelines and Integration

Sowilo operations are pure functions on Rune tensors. They compose naturally
with `|>`, work in batches, and integrate with Kaun training loops.

## Composing Operations

Chain operations with the pipe operator:

<!-- $MDX skip -->
```ocaml
open Sowilo

let process img =
  img
  |> to_float
  |> to_grayscale
  |> gaussian_blur ~sigma:1.0
  |> threshold 0.5

let edges img =
  img
  |> to_float
  |> to_grayscale
  |> canny ~low:0.2 ~high:0.6
```

Since every operation takes a tensor and returns a tensor, you can define
reusable pipeline functions and combine them:

<!-- $MDX skip -->
```ocaml
open Sowilo

let preprocess img =
  img
  |> to_float
  |> resize ~height:256 ~width:256
  |> center_crop ~height:224 ~width:224

let enhance img =
  img
  |> adjust_contrast 1.2
  |> unsharp_mask ~sigma:1.0 ~amount:0.5

let full_pipeline img = img |> preprocess |> enhance
```

## Batch Processing

All operations handle both single images `[H; W; C]` and batches
`[N; H; W; C]`. Stack images into a batch, process in one call:

<!-- $MDX skip -->
```ocaml
open Sowilo

let process_batch paths =
  (* Load and stack into [N; H; W; C] *)
  let images =
    List.map
      (fun p -> Nx_io.load_image p|> to_float)
      paths
  in
  let batch = Rune.stack ~axis:0 images in

  (* All operations broadcast over the batch dimension *)
  let processed =
    batch
    |> resize ~height:224 ~width:224
    |> to_grayscale
    |> gaussian_blur ~sigma:1.0
  in
  processed
```

## Deep Learning Preprocessing

Prepare images for neural networks with standard preprocessing:

<!-- $MDX skip -->
```ocaml
open Sowilo

(* ImageNet preprocessing *)
let imagenet_preprocess img =
  img
  |> to_float
  |> resize ~height:256 ~width:256
  |> center_crop ~height:224 ~width:224
  |> normalize
      ~mean:[0.485; 0.456; 0.406]
      ~std:[0.229; 0.224; 0.225]
```

## Differentiable Augmentation

Since most sowilo operations are differentiable, you can use them as
augmentations inside a training loop and gradients will flow through:

<!-- $MDX skip -->
```ocaml
open Sowilo

(* Differentiable augmentation pipeline *)
let augment img =
  img
  |> adjust_brightness 1.1
  |> adjust_contrast 0.9
  |> adjust_saturation 1.1
  |> gaussian_blur ~sigma:0.3

(* Use in a loss function - gradients flow through augmentation *)
let loss params img label =
  let preprocessed = imagenet_preprocess (augment img) in
  let logits = model params preprocessed in
  cross_entropy logits label

(* Rune.grad differentiates through augment + preprocess + model *)
```

Operations that break the gradient (`canny`, `median_blur`) should not be
used inside differentiable pipelines. All other operations -- blurs, color
adjustments, geometric transforms, morphology, threshold, sobel, scharr,
laplacian -- support `Rune.grad`.

## Integration with Kaun

Use sowilo preprocessing in Kaun data pipelines:

<!-- $MDX skip -->
```ocaml
open Sowilo
open Kaun

let preprocess img =
  img
  |> Sowilo.to_float
  |> Sowilo.resize ~height:224 ~width:224
  |> Sowilo.normalize
      ~mean:[0.485; 0.456; 0.406]
      ~std:[0.229; 0.224; 0.225]

let train_data =
  Data.prepare ~shuffle:rngs ~batch_size:32 (images, labels)
  |> Data.map (fun (x, y) ->
    (preprocess x, fun logits -> Loss.cross_entropy_sparse logits y))
```

## Feature Extraction

Combine edge detection with morphological operations to extract features:

<!-- $MDX skip -->
```ocaml
open Sowilo

let extract_features img =
  let gray = to_grayscale img in

  (* Edge features *)
  let gx, gy = sobel gray in
  let magnitude = Rune.sqrt (Rune.add (Rune.mul gx gx) (Rune.mul gy gy)) in

  (* Morphological features *)
  let kernel = structuring_element Rect (3, 3) in
  let gradient = morphological_gradient ~kernel gray in

  (* Stack as multi-channel feature map *)
  Rune.concatenate ~axis:(-1) [ gray; magnitude; gradient ]
```

## Visualization

Display processing results side by side with Hugin:

<!-- $MDX skip -->
```ocaml
open Sowilo

let visualize_pipeline img =
  let gray = to_grayscale img in
  let blurred = gaussian_blur ~sigma:2.0 gray in
  let edges = canny ~low:0.2 ~high:0.6 gray in

  let fig = Hugin.figure ~width:1200 ~height:400 () in

  let ax1 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:1 fig in
  ignore
    (ax1
    |> Hugin.Plotting.imshow ~data:gray
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Grayscale");

  let ax2 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:2 fig in
  ignore
    (ax2
    |> Hugin.Plotting.imshow ~data:blurred
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Gaussian Blur");

  let ax3 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:3 fig in
  ignore
    (ax3
    |> Hugin.Plotting.imshow ~data:edges
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Canny Edges");

  Hugin.show fig
```

## Color Space Manipulation

Adjust colors through HSV for more precise control:

<!-- $MDX skip -->
```ocaml
open Sowilo

(* Selective color manipulation via HSV *)
let make_warmer img =
  let hsv = rgb_to_hsv img in
  (* Shift hue slightly toward warm tones *)
  let adjusted = adjust_hue 0.02 img in
  (* Boost saturation *)
  adjust_saturation 1.2 adjusted

(* Grayscale with tint *)
let sepia img =
  img
  |> to_grayscale
  |> fun gray ->
     (* Expand back to 3 channels and tint *)
     let rgb = Rune.concatenate ~axis:(-1) [ gray; gray; gray ] in
     adjust_saturation 0.3 (adjust_hue 0.05 rgb)
```
