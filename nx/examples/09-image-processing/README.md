# `09-image-processing`

Load, transform, and save images as arrays — convolutions, pooling, and pixel
math. This example creates a synthetic grayscale image, blurs it, detects edges
with Sobel filters, and downsamples with max pooling.

```bash
dune exec nx/examples/09-image-processing/main.exe
```

## What You'll Learn

- Creating synthetic images with `init` and pixel math
- Applying 2D convolution with `correlate2d` (NCHW format)
- Gaussian blur with a 3x3 kernel
- Sobel edge detection (horizontal + vertical gradients)
- Downsampling with `max_pool2d`
- Converting between `UInt8` and `Float32` for computation
- Saving arrays as PNG files with `Nx_io.save_image`

## Key Functions

| Function                                      | Purpose                                            |
| --------------------------------------------- | -------------------------------------------------- |
| `init UInt8 shape f`                          | Create an image by computing each pixel            |
| `correlate2d ~padding_mode:\`Same img kernel` | 2D convolution (expects NCHW)                      |
| `max_pool2d ~kernel_size ~stride img`         | Downsample by taking max in each window            |
| `cast Float32 t`                              | Convert dtype for floating-point operations        |
| `clamp ~min ~max t`                           | Clamp values to a valid pixel range                |
| `contiguous t`                                | Ensure contiguous memory layout (required for I/O) |
| `Nx_io.save_image path t`                     | Save a 2D (HxW) array as a grayscale PNG           |

## Output Walkthrough

### Synthetic image

A 64x64 horizontal gradient with a bright rectangle in the center:

```ocaml
let img = init UInt8 [| h; w |] (fun idx ->
    let y = idx.(0) and x = idx.(1) in
    let base = x * 255 / (w - 1) in
    if y >= 16 && y < 48 && x >= 16 && x < 48 then 220 else base)
```

### NCHW format

Convolution operations expect 4D tensors in NCHW format (batch, channels,
height, width). Convert with:

```ocaml
let img_f = cast Float32 img |> contiguous |> reshape [| 1; 1; h; w |]
```

### Gaussian blur

A 3x3 kernel with weights summing to 1, giving more weight to the center:

```ocaml
let blur_kernel = create Float32 [| 1; 1; 3; 3 |]
  [| 1./16.; 2./16.; 1./16.;
     2./16.; 4./16.; 2./16.;
     1./16.; 2./16.; 1./16. |]
```

### Sobel edge detection

Combines horizontal and vertical gradient magnitudes:

```ocaml
let gx = correlate2d ~padding_mode:`Same img_f sobel_x in
let gy = correlate2d ~padding_mode:`Same img_f sobel_y in
let edges = sqrt (add (mul gx gx) (mul gy gy))
```

### Max pooling

2x downsampling by taking the maximum in each 2x2 window:

```
Saved: pooled.png (64x64 -> 32x32)
```

## Output Files

Running this example creates four PNG files in the current directory:

| File           | Description                    |
| -------------- | ------------------------------ |
| `gradient.png` | Original synthetic image       |
| `blurred.png`  | After Gaussian blur            |
| `edges.png`    | Sobel edge detection result    |
| `pooled.png`   | 2x downsampled via max pooling |

## Try It

1. Replace the blur kernel with a sharpening kernel:
   `[| 0.; -1.; 0.; -1.; 5.; -1.; 0.; -1.; 0. |]`
2. Try a larger pooling window (`4, 4`) and observe the effect on image size
   and detail.
3. Chain blur and edge detection: blur first to reduce noise, then apply Sobel.

## Next Steps

Continue to [10-data-pipeline](../10-data-pipeline/) to see a complete data
preparation pipeline with the Iris dataset.
