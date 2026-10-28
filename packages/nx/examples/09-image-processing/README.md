# `09-image-processing`

Load, transform, and save images as arrays: convolutions, pooling, and pixel
math. This example creates a synthetic grayscale image, blurs it, detects edges
with Sobel filters, and downsamples with max pooling.

```bash
dune exec packages/nx/examples/09-image-processing/main.exe
```

## What You'll Learn

- Creating synthetic images with `init` and pixel math
- Applying 2D cross-correlation with `correlate`
- Gaussian blur with a 3x3 kernel
- Sobel edge detection (horizontal + vertical gradients)
- Downsampling by max pooling, built from `sliding_window` and `max`
- Converting between `uint8` and `float32` for computation
- Saving arrays as PNG files with `Nx_io.save_png`

## Key Functions

| Function                                      | Purpose                                         |
| --------------------------------------------- | ----------------------------------------------- |
| `init uint8 shape f`                          | Create an image by computing each pixel         |
| ``correlate ~padding:`Same img kernel``       | Cross-correlation over the kernel's axes        |
| `sliding_window ~axis ~window ~step t`        | View an axis as windows, one every `step`       |
| `max ~axes t`                                 | Maximum over the window axes                    |
| `cast float32 t`                              | Convert dtype for floating-point operations     |
| `clamp ~min ~max t`                           | Clamp values to a valid pixel range             |
| `unsqueeze ~axes:[ 2 ] t`                     | Add the channel axis an image needs             |
| `Nx_io.save_png path t`                       | Save an `[h; w; c]` `uint8` array as a PNG      |

## Output Walkthrough

### Synthetic image

A 64x64 horizontal gradient with a bright rectangle in the center:

```ocaml
let img = Nx.init Nx.uint8 [| h; w |] (fun idx ->
    let y = idx.(0) and x = idx.(1) in
    let base = x * 255 / (w - 1) in
    if y >= 16 && y < 48 && x >= 16 && x < 48 then 220 else base)
```

### Cross-correlation

`correlate` slides the kernel over the last axes of the image, as many as the
kernel has, so a `[3; 3]` kernel works on the `[h; w]` image directly.
`` ~padding:`Same `` keeps the image's size, reading zeros past its edges.

### Gaussian blur

A 3x3 kernel with weights summing to 1, giving more weight to the center:

```ocaml
let blur_kernel = Nx.create Nx.float32 [| 3; 3 |]
  [| 1./16.; 2./16.; 1./16.;
     2./16.; 4./16.; 2./16.;
     1./16.; 2./16.; 1./16. |]
```

### Sobel edge detection

Combines horizontal and vertical gradient magnitudes:

```ocaml
let gx = Nx.correlate ~padding:`Same img_f sobel_x in
let gy = Nx.correlate ~padding:`Same img_f sobel_y in
let edges = Nx.(Infix.(sqrt ((gx * gx) + (gy * gy))))
```

### Max pooling

2x downsampling by taking the maximum in each 2x2 block. `sliding_window`
views an axis as windows of 2 elements, one every 2, appending an axis for
the window: `[64; 64]` becomes `[32; 64; 2]`, then `[32; 32; 2; 2]`. The
maximum over the two window axes is each block's:

```ocaml
let pooled =
  img_f
  |> Nx.sliding_window ~axis:0 ~window:2 ~step:2
  |> Nx.sliding_window ~axis:1 ~window:2 ~step:2
  |> Nx.max ~axes:[ 2; 3 ]
```

```
Pooled: 64x64 -> 32x32
```

### Saving

An image is a `uint8` array of shape `[h; w; c]`, `c` the number of channels.
A grey image has one channel, so each result is clamped to [0, 255], cast to
`uint8` and given a channel axis before `save_png`:

```ocaml
let pixels = Nx.clamp ~min:0.0 ~max:255.0 x |> Nx.cast Nx.uint8 in
Nx_io.save_png path (Nx.unsqueeze ~axes:[ 2 ] pixels)
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
2. Try a larger pooling window (`~window:4 ~step:4`) and observe the effect on
   image size and detail.
3. Chain blur and edge detection: blur first to reduce noise, then apply Sobel.

## Next Steps

You've completed the Nx examples! For machine learning workflows, see the
[kaun examples](../../../kaun/doc/index.md).
