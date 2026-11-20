# Changelog

All notable changes to Mynd are documented in this file.

## Unreleased

- Add `Box.transform`, which moves boxes by a transform. A box that meets
  `d = 0`, whose image is unbounded, moves to NaN, and an empty box to
  `0 0 0 0`. The guide page `doc/annotations.md` moves masks and keypoints.
- Add `Box.giou`, generalized IoU paired as `Box.iou` pairs boxes, with
  `c - u` clamped at 0 so that rounding never puts it above `iou`.
- New library `mynd.coco`: COCO's files through jsont, polygon masks by
  `Mynd_coco.runs`, and `evaluate_boxes`, `evaluate_masks` and `summary`,
  whose tables equal pycocotools 2.0.11's bit for bit on valid data. A mask
  result's area range reads its pixel count, where pycocotools reads the
  `bbox` a results file holds beside it.
- Add ten examples (`examples/`) that run with the test suite on a CC0
  photograph, from an image's values to masks, detections and a learned
  warp. sowilo's blurs are in `05-filters`, its Sobel and Canny in
  `06-edges`, and its threshold and morphology in `07-morphology`.
- mynd replaces sowilo, which raven no longer ships. Each sowilo value
  becomes a mynd value, a composition on a page of the guide (`doc/`) or
  an nx call, in the table below. Every replacement takes any float dtype
  and any number of batch axes, and a parameter held in an array, such as
  a blur's deviation, has a derivative. The rows' images are `[h; w; c]`;
  `sigma` and `mean`, `std` are arrays, `[]` and `[c]`. Results change:
  - `convert` scales each dtype by its own largest code, where `to_float`
    divided every dtype by 255, floats too. `convert Nx.uint8` rounds to
    the nearest code, where `to_uint8` truncated, so a code whose scaled
    value's fraction is above a half moves up by one.
  - `Sample.resize` reads cell centres, where sowilo's `Bilinear` aligned
    the corner pixels, and band-limits a shrink. It takes floats, where
    sowilo took every dtype and truncated back into it. `Sample.nearest` at
    scaled centres can take the other pixel where a centre falls on the edge
    between two pixels, which sowilo settled by Pillow's running sum.
  - Filters, gradients and blurs take an `~edge` rule; sowilo's read zero
    outside the image, which is `` `Zero ``. A Gaussian's radius is
    explicit: sowilo's default of `2⌈3σ⌉ + 1` taps is `~radius:⌈3σ⌉`, and
    its fixed tables for a `sigma` of `0` or less are gone.
  - The guide's contrast blends each image with its own mean luma, where
    `adjust_contrast` blended each channel with its mean over the whole
    batch; its saturation blends with each pixel's luma, where
    `adjust_saturation` scaled HSV saturation. `Colour.to_hsv` adds no
    `1e-7` to the chroma and the value, which biased hue and saturation.
  - `Filter.gradient` is in values per pixel: sowilo's `sobel` was 8 times
    `` `Sobel ``'s (2 times `` `Central ``'s at `~ksize:1`, 128 and 2048
    times the guide's wider stencils at 5 and 7) and `scharr` 32 times
    `` `Scharr ``'s. Canny's thresholds are strengths in those units:
    with `` `Sobel `` in the guide's composition, sowilo's divided by 8
    compare the same strengths. It thins by interpolation along the
    gradient, where `canny` swapped its two diagonals, and keeps weak
    pixels joined to a strong one through any path of 8-connected weak
    pixels, where `canny` kept those within two pixels of one.
  - `Filter.dilate` reads the footprint reflected about its origin offset,
    so that `dilate fp (erode fp x)` is an opening; sowilo's read it
    unreflected. They agree on a footprint symmetric about its centre.
    The guide's `disk 2` keeps one pixel in its first and last rows, where
    sowilo's 5×5 `Ellipse` kept three.

  | sowilo | mynd |
  |---|---|
  | `to_float x` | `convert Nx.float32 x` |
  | `to_uint8 x` | `convert Nx.uint8 x` |
  | `normalize ~mean ~std x` | `Nx.div (Nx.sub x mean) std` |
  | `threshold t x` | `Nx.greater_s x t` |
  | `to_grayscale x` | `Colour.luma x`, of exactly three channels: split alpha off first with `Nx.split_at` |
  | `rgb_to_hsv`, `hsv_to_rgb` | `Colour.to_hsv`, `Colour.of_hsv`, hue in turns as before |
  | `adjust_brightness`, `adjust_contrast`, `adjust_saturation` | `doc/colour.md`, Brightness, contrast and saturation |
  | `adjust_hue d x` | `doc/colour.md`, Hue |
  | `adjust_gamma g x` | `Nx.pow_s x g` |
  | `invert x` | `Nx.rsub_s 1. x` |
  | `resize ~interpolation:Bilinear` | `` Sample.resize ~edge:`Replicate `Triangle ``; aligned corners are `` Sample.at ~edge:`Replicate `Triangle `` at `(j (w - 1) / (w' - 1) + ½, i (h - 1) / (h' - 1) + ½)` |
  | `resize ~interpolation:Nearest` | `` Sample.nearest ~edge:`Replicate `` at `Sample.centres` times `(w / w', h / h')` |
  | `crop ~y ~x ~height ~width` | `Nx.slice [ R (y, y + height); R (x, x + width) ]` |
  | `center_crop` | that slice from `y = (h - height) / 2`, `x = (w - width) / 2` |
  | `hflip`, `vflip` | `Nx.flip ~axes:[ -2 ]`, `Nx.flip ~axes:[ -3 ]` |
  | `rotate90 x` | `Nx.flip ~axes:[ -3 ] (Nx.swapaxes (-3) (-2) x)` |
  | `pad ~value (t, b, l, r) x` | `Nx.pad [\| (t, b); (l, r); (0, 0) \|] value x`, or `Nx.extend` by a rule |
  | `gaussian_blur ~sigma ~ksize` | `` Filter.gaussian ~edge:`Zero ~radius:(ksize / 2) sigma `` |
  | `box_blur` | `doc/filters.md`, Box blur |
  | `median_blur ~ksize` | `` Filter.median ~edge:`Zero (ksize, ksize) `` |
  | `filter2d k` | `` Filter.correlate ~edge:`Zero k `` |
  | `unsharp_mask` | `doc/filters.md`, Unsharp mask |
  | `laplacian` | `doc/filters.md`, Laplacian, at sowilo's default `~ksize:1` |
  | `structuring_element` | `doc/morphology.md`, Footprints |
  | `erode ~kernel` | `Filter.erode kernel` |
  | `dilate ~kernel` | `Filter.dilate (Nx.flip kernel)`, a row or column of `false` put before each even axis |
  | `opening ~kernel`, `closing ~kernel` | `doc/morphology.md`, Openings and closings: the opening by `kernel`, the closing by `Nx.flip kernel` for odd extents |
  | `morphological_gradient` | `doc/morphology.md`, Top-hats and the morphological gradient |
  | `sobel`, `scharr` | `` Filter.gradient ~edge:`Zero `Sobel ``, `` `Scharr ``; `doc/edges.md`, Gradients, for apertures 5 and 7 |
  | `canny` | `doc/edges.md`, Canny's edges |

- Add morphology: `Filter.erode` and `Filter.dilate` by host boolean
  footprints of any extents, reading no extension rule, with dilation by
  Minkowski addition so that `dilate fp (erode fp x)` is an opening for
  every footprint; ties share the derivative equally. Add
  `Mask.reconstruct`, the components of a mask that hold a marker pixel,
  in `Mask.components`' rounds.
- `Mask`'s doc gives areas per mask of a batch, by a scatter along each
  mask's flattened labels.
- Add gradients: `Filter.gradient`, the pair `(gx, gy)` in values per
  pixel under the `Central`, `Sobel` and `Scharr` stencils, exact on ramps
  and under flips and transposes; `Filter.gradient_maxima`, Canny's
  thinning by interpolation along the gradient, keeping at most one pixel
  of a tie.
- Add filtering: `Filter.windows`, a view of each pixel's window under an
  extension rule; `Filter.correlate` by weights paired with the images'
  batch axes; `Filter.gaussian`, the sampled Gaussian with a radius and a
  deviation held in data; `Filter.median`, the lower median in nx's sort
  order for every dtype.
- Add colour models: `Colour.luma` (BT.601, on encoded values) and
  `Colour.luminance` (CIE Y of linear sRGB), exact on greys;
  `Colour.to_hsv` and `Colour.of_hsv`, Smith's hexcone with hue in turns,
  whose round trip has the identity as derivative at tied channels too;
  `Colour.linear_to_lab` and `Colour.lab_to_linear`, CIE L*a*b* relative
  to D65 with CSS Color 4's rational matrices.
- New library `mynd`, images over nx: `convert` between dtypes by Vulkan's
  normalized fixed-point rules, `srgb_to_linear` and `linear_to_srgb`,
  `premultiply` and `unpremultiply`, `Sample.at` and `Sample.nearest` at
  positions under box, triangle, Keys' cubic and Lanczos kernels with an
  edge rule, `Transform`, band-limited `Sample.resize` and its
  `Sample.resize_weights`, `Box`, `Mask` and COCO's `Rle`,
  `Mask.components` and `Box.nms`. Each is a composition of nx operations
  that rune differentiates, batches and compiles.
