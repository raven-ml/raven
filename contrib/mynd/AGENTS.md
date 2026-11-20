# mynd

Images, positions in them and their annotations over nx: values and their
conversion, sampling at positions, transforms, resizing, boxes, masks,
COCO's run-length encoding, connected components and non-maximum
suppression. It is a project of its own in `contrib/`: its `dune-project`
declares the opam package `mynd`, and it keeps its own `CHANGES.md`. It
builds and tests against raven's main, releases on its own schedule and
carries no API stability guarantee. It replaces sowilo, raven's earlier
vision package, removed once mynd covered its surface.

- Changes go in `contrib/mynd/CHANGES.md`.
- The same rules as raven's libraries: `docs/dev/design.md`, AGENTS.md's code
  style, and the docs follow `docs/dev/documentation.md`.
- Each golden in `test/` names its generator in `test/gen/`, which runs
  through `uv run --locked` in that directory's pinned environment and
  writes the golden byte for byte.

## Boundaries

- mynd depends on the public libraries of core packages alone: on nx, and on
  rune for `Rune.iterate` and `Rune.custom_jvp`. It never calls `Rune.jit`:
  the caller compiles. The one exception, approved by the maintainer, is
  jsont for `mynd.coco`'s codecs of COCO's files; the core library `mynd`
  never depends on it.
- mynd never depends on alda, raven's signal library. Its kernels and its
  resizing weights are its own; what both need, such as `Nx.sinc`, is built
  in nx.
- An image is a plain `Nx.t` `[…; h; w; c]`, channel axis always present;
  a mask is a boolean image `[…; h; w; 1]`. No image record, colour space
  or alpha flag: encoding and alpha are the caller's facts.
- A capability mynd lacks below it (an extension rule, a gather, a stable
  form of a function) is built in nx, never worked around here. The edge
  rule's index fold is `Nx.extension_index`.
- Run IoU, run unions, polygons and crowd IoU belong to the COCO evaluator,
  not here. Pillow's 8-bit resize is a composition over
  `Sample.resize_weights` at its call site.

## Decisions

- `convert` follows Vulkan's normalized fixed-point conversions: UNORM
  for unsigned, SNORM for signed codes; integer pairs are exact through
  the repeating binary expansion of `k / (2ˢ - 1)`, in uint64.
- One coordinate system: pixel `(i, j)` is the cell
  `[j, j + 1) × [i, i + 1)`; nearest reads `⌊u⌋`, a cell edge the cell it
  starts.
- Kernels are data with a whole `2h`, so `Sample.at` reads exactly `2h` taps
  an axis. A tap's argument is `f - (t - ⌊u⌋ + ½)` with `f = u - ⌊u⌋` exact,
  one rounding, so the derivative at a kink is the right-hand one. Each
  axis's weights are divided by their sum: every read is a weighted mean,
  under every kernel and edge.
- `Sample.resize` is a gather and a sum per tap slot of a host table,
  columns then rows, never a dense matrix, so a pixel reaches only the new
  pixels that weigh it. The table's weights come from exact ratios at
  float64. A trace stages every nx operation, so the table is built from
  OCaml floats and integers, the kernels' formulas evaluated there too, and
  the weights are stored straight into the pass's dtype. The fold stays
  `Nx.extension_index`, so under a trace each slot's fold, and its `where`
  for the padding, are staged: a compiled resize runs them on the host at
  every call and copies the indices to the image's device.
- `Mask.components` is Liu and Tarjan's R with two shortcuts a round; its
  `~max` is their proven bound. `Box.nms` is the Jacobi fixed point of the
  greedy rule; its threshold compares exactly with the OCaml float.
- `Filter.gradient` is in values per pixel: a difference of `-½ 0 ½` after
  weights across that sum to one, so `Sobel` and `Scharr` are the libraries'
  kernels over 8 and 32. Outer weights multiply the sum of their two pixels,
  so flips and transposes are exact on images. `gradient_maxima` is Canny's
  ring interpolation, ties kept on the side the gradient points away from,
  and reads nothing outside the image. Canny's detector is a guide
  composition: with `Filter.gaussian ~edge:`Zero` over the blur of ones,
  `Sobel` under `Symmetric_half`, the border ring dropped and thresholds
  `>=` over 8, it equals scikit-image's `canny` at its defaults
  (test_gradient).
- Morphology reads no extension rule (outside offsets left out, the dtype's
  extreme where none is inside), and `dilate` is Minkowski addition, the
  convention under which `dilate fp (erode fp x)` is an opening for every
  footprint. OpenCV's and scikit-image 1.x's `dilate` read erosion's
  offsets, and scikit-image anchors an even axis one offset earlier: their
  parity is a footprint conversion (test_morphology), and their opening of a
  footprint asymmetric about its origin offset is no opening. The tie
  derivative counts and sums at float32 for the formats narrower than 32
  bits, as rune's `Nx.min` counts in the carrier. Complex images raise.
- `Mask.reconstruct` is the components holding a marker pixel, in
  `components`' rounds, never dilation iterated to stability, which takes up
  to `h w / 2` rounds on a serpentine. Labels index each mask's own pixels,
  so per-label sums scatter along the last axis of `[…; h w]`; a flat
  `reduce_segments` over a batch merges labels across masks.
- `Mask.components_rounds` and `Box.nms_rounds`, behind their modules'
  stops, give the tests their step counts.

## Traps

- `Nx.take` reads zero outside `[0, d)` per axis only: a `-1` inside a
  flat index `i w + j` names another pixel.
- `where` passes a NaN derivative from the branch not taken unless that
  branch's argument is made safe first (the transfer functions, IoU's
  empty union, `unpremultiply`, a weight sum of `0`).
- torch's `grid_sample` unnormalises its grid, which moves a position by
  an ulp unless it is dyadic in the image's size: compare derivatives at
  centres only on such positions.
- torch 2.9's antialiased `interpolate` reads an image one column wide as
  constant: compare one axis along the width.

## Parity

Facts for reproducing another library's numbers in a test or a port.
The guide pages under `doc/` explain mynd alone.

- Edge rules: `` `Zero `` is OpenCV's `BORDER_CONSTANT`, scipy.ndimage's
  `constant` and torch's `constant` padding; `` `Replicate `` is
  `BORDER_REPLICATE`, `nearest` and `replicate`; `` `Symmetric_whole ``
  is `BORDER_REFLECT_101` (OpenCV's default), `mirror` and `reflect`;
  `` `Symmetric_half `` is `BORDER_REFLECT` and `reflect` (scipy's
  default), with no torch mode; `` `Periodic `` is `BORDER_WRAP`, `wrap`
  and `circular`. OpenCV's `filter2D` takes no `BORDER_WRAP`.
- A window's offset `k / 2` is the pixel itself, which is scipy.ndimage's
  `origin` 0 and OpenCV's anchor `(-1, -1)` at every extent.
  `median_filter` takes rank `n / 2`, the upper median for an even `n`;
  `rank_filter` at `(n - 1) / 2` is `Filter.median`'s lower one.
- torchvision 0.24's colour adjustments (`ColorJitter`): its grey weighs
  `0.2989 r + 0.587 g + 0.114 b`, which sums to `0.9999`; each blend is
  `f x + (1 - f) r`, clamped to `[0, 1]` after every adjustment, and
  differs from `Nx.lerp` by rounding; contrast takes the mean of the grey
  over the last three axes, flooring an integer image's grey first, and an
  image of one channel is its own grey; the hue factor is in turns, in
  `[-0.5, 0.5]`; an integer result is truncated toward zero;
  `ColorJitter` applies its four adjustments in an order drawn for each
  call.
- torchvision's `GaussianBlur` samples the Gaussian divided by its sum, as
  `Filter.gaussian` does, with `kernel_size = 2 radius + 1`, and draws its
  deviation uniformly from `(0.1, 2)` by default.
- mynd's gradient strengths are in values per pixel on images in
  `[0, 1]`. Another library's strength is a multiple of mynd's:

| Library | Its strength in mynd's units |
|---|---|
| scipy `ndimage.sobel`, scikit-image raw Sobel | 8 × |
| OpenCV `Sobel` of aperture `k` = 3, 5, 7 on uint8 codes | `2^(2k − 3) · 255` ×, so 8, 128 and 2048 times 255 |
| OpenCV `Scharr` on uint8 codes | 32 · 255 × |
| kornia `canny` (raw Sobel, L2, `eps = 1e-6` under the root) | 8 × the input's full scale; a flat pixel reads `1.25e-4` |

- OpenCV's `Canny` takes thresholds in the raw units of its aperture (its
  `/16` at aperture 7 cancels its own Sobel scale), uses L1 strength
  `|gx| + |gy|` unless `L2gradient` (no factor converts L1), applies no
  blur, thins by 45° sectors with index-order ties, and reads its Sobel
  under `BORDER_REPLICATE` with strength 0 outside. Apertures 5 and 7 are
  one or two `¼ ½ ¼` blurs then `Sobel`, times 128 and 2048.
- scikit-image's `canny` at its defaults uses thresholds `0.1` and `0.2`
  of raw Sobel on the `[0, 1]` image whatever the dtype; given thresholds
  are divided by the dtype's maximum first; `use_quantiles` takes
  quantiles of the strength over every pixel. Its blur is `mode='constant'`
  divided by the blur of ones (plus float64's epsilon), radius
  `⌊4σ + ½⌋`; Sobel under `reflect` (`Symmetric_half`); the border ring is
  never an edge; thresholds compare with `>=`; it keeps both pixels of a
  tie. test_gradient composes it exactly.
- MATLAB's thresholds are fractions of a strength its page leaves
  undefined; a scalar sets low to `0.4 ·` high.
- OpenCV's `Canny` replicates, where its filters default to
  `BORDER_REFLECT_101`.
- OpenCV `erode fp` is `erode fp`. OpenCV `dilate fp` reads erosion's
  offsets: it is `dilate (reflect_pad fp)`, `reflect_pad` flipping `fp` and
  putting a row or column of `false` before each even axis. A footprint with
  a `true` offset in the first row or column of an even axis, a full
  rectangle included, is asymmetric about offset `k / 2`, so OpenCV's
  dilation by `np.ones((2, 2))` differs from mynd's by a pixel. Its
  `MORPH_OPEN` of such a footprint is no opening; do not reflect the
  footprint in mynd's opening.
- scikit-image 1.x anchors an even axis one offset earlier (`pad_front`, a
  row of `false` before it): `erosion fp` is `erode (pad_front fp)`,
  `dilation fp` is `dilate (reflect_pad (pad_front fp))`, with
  `mode="ignore"`. scipy's `grey_dilation` and kornia's dilation are
  mynd's under the reduction's identity outside.
- Other border rules equal mynd's left-out offsets for some footprints:
  `nearest` when each row and column of offsets is one run through the
  origin's row or column; `reflect` when also each run's ends `a ≤ 0 ≤ b`
  have `|a + b| ≤ 1`; `mirror` when each run is centred.
- scipy's `binary_fill_holes` is `not (reconstruct c ~marker:frame
  (not m))` with `c` its structure's connectivity (test_mask).
- SAM 2's mask post-processing labels with 8-connectivity, removes
  components and fills background components of area `<=` its limit, the
  latter touching the border included, both from the same mask.
