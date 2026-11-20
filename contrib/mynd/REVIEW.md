# Reviewing mynd

## Design

- A loop or kernel of mynd's own where nx operations compose the result;
  rune can't differentiate, batch or compile what mynd hides. A capability
  nx lacks is built in nx.
- A dependency on alda, or on anything but nx and rune, except jsont in
  `mynd.coco` (approved for that library only).
- An image record, a colour space or an alpha flag where the caller's
  facts and an explicit function serve.
- A call to `Rune.jit`: the caller compiles.

## Semantics

References: the specifications and programs each module of `mynd.mli`
cites (Vulkan's fixed-point conversions, CSS Color 4, Keys, Liu and
Tarjan, pycocotools, torchvision), and the generator of each golden under
`test/gen/`.

Failure here is a pixel off by half a cell or a weight that does not sum
to one: a centre at `j` where it lies at `j + ½`, an edge that reads the
wrong pixel, a flat image that does not stay flat, a mask that loses a
pixel to a cast.

- Positions at cell edges, centres, just outside the image and far away,
  non-finite ones, in every float format.
- Every edge rule on images of one row or column, and none.
- Resizes that shrink and enlarge, by ratios with and without a short
  binary form, against torch and Pillow.
- Derivatives in positions at kinks and where a weight sum is `0`.
