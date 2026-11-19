# mynd

Images, positions in them and their annotations over nx: values and their
conversion, sampling at positions, transforms, resizing, boxes, masks,
COCO's run-length encoding, connected components and non-maximum
suppression. It is a project of its own in `contrib/`: its `dune-project`
declares the opam package `mynd`, and it keeps its own `CHANGES.md`. It
builds and tests against raven's main, releases on its own schedule and
carries no API stability guarantee. It replaces `contrib/sowilo`, which
stays until mynd covers its surface.

- Changes go in `contrib/mynd/CHANGES.md`.
- The same rules as raven's libraries: `docs/dev/design.md`, AGENTS.md's code
  style, and the docs follow `docs/dev/documentation.md`.
- Each golden in `test/` names its generator in `test/gen/`, which runs
  through `uv run --locked` in that directory's pinned environment and
  writes the golden byte for byte.

## Boundaries

- mynd depends on the public libraries of core packages alone: on nx, and on
  rune for `Rune.iterate`. It never calls `Rune.jit`: the caller compiles.
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
  not here. Pillow's 8-bit resize is a composition over `resize_weights` at
  its call site.

## Decisions

- `convert` follows Vulkan's normalized fixed-point conversions: UNORM
  for unsigned, SNORM for signed codes; integer pairs are exact through
  the repeating binary expansion of `k / (2ˢ - 1)`, in uint64.
- One coordinate system: pixel `(i, j)` is the cell
  `[j, j + 1) × [i, i + 1)`; nearest reads `⌊u⌋`, a cell edge the cell it
  starts.
- Kernels are data with a whole `2h`, so `sample` reads exactly `2h` taps
  an axis. A tap's argument is `f - (t - ⌊u⌋ + ½)` with `f = u - ⌊u⌋`
  exact, one rounding, so the derivative at a kink is the right-hand one.
  Each axis's weights are divided by their sum: every read is a weighted
  mean, under every kernel and edge.
- `resize` is a gather and a sum per tap slot of a host table, columns then
  rows, never a dense matrix, so a pixel reaches only the new pixels that
  weigh it. The table's weights come from exact ratios at float64. A trace
  stages every nx operation, so the table is built from OCaml floats and
  integers, the kernels' formulas evaluated there too, and the weights are
  stored straight into the pass's dtype. The fold stays
  `Nx.extension_index`, so under a trace each slot's fold, and its `where`
  for the padding, are staged: a compiled resize runs them on the host at
  every call and copies the indices to the image's device.
- `components` is Liu and Tarjan's R with two shortcuts a round; its
  `~max` is their proven bound. `nms` is the Jacobi fixed point of the
  greedy rule; its threshold compares exactly with the OCaml float.
- `components_rounds` and `nms_rounds`, behind the `.mli`'s stop, give the
  tests their step counts.

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
- A window's tap `k / 2` is the pixel itself, which is scipy.ndimage's
  `origin` 0 and OpenCV's anchor `(-1, -1)` at every extent.
  `median_filter` takes rank `n / 2`, the upper median for an even `n`;
  `rank_filter` at `(n - 1) / 2` is `median_blur`'s lower one.
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
  `gaussian_blur` does, with `kernel_size = 2 radius + 1`, and draws its
  deviation uniformly from `(0.1, 2)` by default.
