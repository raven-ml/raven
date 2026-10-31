# Sowilo

Differentiable computer vision on Nx tensors.

Sowilo provides image processing operations expressed purely through Nx
tensor operations. Filters, edge detectors, morphological operations, color
transforms, and geometric transforms -- all compatible with `Rune.grad'`
and `Rune.vmap'`.

## Image Conventions

Images are `Nx.t` tensors with channels-last layout:

- **Single image**: `[H; W; C]` (height, width, channels)
- **Batch**: `[N; H; W; C]` (batch, height, width, channels)
- **Grayscale**: C = 1, **RGB**: C = 3, **RGBA**: C = 4

Operations expect float32 tensors with values in [0, 1]. Use `to_float` to
convert from uint8 and `to_uint8` to convert back.

An image lies on any device set, and an operation computes where it lies:
its kernels and constants are made beside the image, and its result is on
the image's set, as the Device sets section of
[`sowilo.mli`](../lib/sowilo.mli) states.

## What's Included

- **Type conversion**: `to_float`, `to_uint8`, `normalize`, `threshold`
- **Color**: `to_grayscale`, `rgb_to_hsv`, `hsv_to_rgb`, brightness, contrast, saturation, hue, gamma, `invert`
- **Geometric transforms**: `resize`, `crop`, `center_crop`, `hflip`, `vflip`, `rotate90`, `pad`
- **Spatial filters**: `gaussian_blur`, `box_blur`, `median_blur`, `filter2d`, `unsharp_mask`
- **Morphology**: `structuring_element`, `erode`, `dilate`, `opening`, `closing`, `morphological_gradient`
- **Edge detection**: `sobel`, `scharr`, `laplacian`, `canny`

## Quick Start

<!-- $MDX skip -->
```ocaml
open Sowilo

let () =
  (* Load image and convert to float32 [0, 1] *)
  let img = Nx_io.load_image "photo.png" |> to_float in

  (* Process: grayscale, blur, edge detection *)
  let edges =
    img
    |> to_grayscale
    |> gaussian_blur ~sigma:1.0
    |> canny ~low:0.2 ~high:0.6
  in

  (* Save the boolean edge mask as an image *)
  Nx_io.save_png "edges.png" (to_uint8 (Nx.cast Nx.float32 edges))
```

## Learn More

- [Getting Started](01-getting-started.md) -- installation, image conventions, first pipeline
- [Operations Reference](02-operations.md) -- every operation with examples
- [Pipelines and Integration](03-pipelines.md) -- composing pipelines, batch processing, deep learning integration
- [Examples](https://github.com/raven-ml/raven/tree/main/sowilo/examples) -- complete image processing examples
