# `07-morphology`

A mask of the photograph's light pixels, opened and closed by a disk,
split into its 8-connected pieces, and cleaned of the small ones.

```bash
dune exec ./main.exe
```

- `Nx.greater_s` of `Mynd.Colour.luma` at 0.5 gives a mask, a boolean
  image of one channel.
- `Mynd.Filter.erode` and `Mynd.Filter.dilate` by a disk footprint make
  the opening, which removes the light details the disk does not fit in,
  and the closing, which fills the dark ones. The footprint, the opening
  and the closing are those of [`doc/morphology.md`](../../doc/morphology.md).
- `Mynd.Mask.components` labels each piece by the least index of its
  pixels. Counting each label's pixels gives every pixel its piece's
  area (Areas and small objects in the same page), which removes the
  small pieces and finds the largest.
- `Mynd.Box.of_mask` gives the largest piece's box, `[x₀ y₀ x₁ y₁]` in
  pixel edges.

It prints the pixels and pieces of the mask, its opening, its closing and
the mask without small pieces, and the largest piece's area and box.
`morphology.png` shows the mask, its opening and the mask without small
pieces.

Margins: every pixel's luma lies at least `2.7e-4` from 0.5, and every
later step computes on booleans and integers.
