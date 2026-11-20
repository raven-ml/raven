# `04-warps`

Turn the photograph by 30° about its centre, and move labels drawn on the
cat's eyes and nose, and their boxes, with it.

```bash
dune exec ./main.exe
```

- `Mynd.Transform.rotation`, `translation` and `Nx.matmul` make the
  transform `m` that turns positions about the image's centre.
- A warp reads, at each of its own pixels' centres
  (`Mynd.Sample.centres`), the input at the position that maps there:
  the centres mapped by `Nx.inv m` with `Mynd.Transform.apply` (the
  preamble of `Sample` in [`mynd.mli`](../../lib/mynd.mli)).
- `Mynd.Sample.at` reads the image there under Keys' cubic; past the
  border it reads black (`` `Zero ``).
- `Mynd.Sample.nearest` reads the labels at the same positions. It reads
  the pixel whose cell holds each position, so every value it gives is a
  label.
- `Mynd.Box.of_mask` gives each label's box before and after. A box
  itself moves by its corners: mapped by `m`, their hull holds the turned
  label's centres, and the turned label's box lies within that hull grown
  by half a pixel, since a pixel's cell reaches half a pixel past its
  centre.

It prints each label's pixels before and after, its box, the hull of its
corners after the turn, its box after, and whether that box lies within
the hull grown by half a pixel. `warps.png` draws the turned photograph
with the turned labels lightened, their boxes in white and the hulls in
yellow.

Margins: every position that decides a labelled pixel lies at least
`4.8e-3` pixels from a cell's edge, and every pixel's centre at least
`6e-4` from an ellipse in its squared normalised radius. The hulls lie at
least `3.2e-3` from a change of their last printed digit.
