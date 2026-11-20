# `06-edges`

The gradient of the grey photograph, and Canny's edges at two pairs of
thresholds.

```bash
dune exec ./main.exe
```

- `Mynd.Filter.gaussian` blurs the grey photograph by a deviation of 1.5
  pixels, and `Mynd.Filter.gradient` with `` `Sobel `` gives its
  gradient `(gx, gy)`, in values per pixel. The strength is
  `Nx.hypot gx gy`. A step from black to white between two columns,
  blurred by 1.5 pixels, has strength `(g(0) + g(1)) / 2` beside it,
  `g` the blur's weights, about 0.24, so the photograph's largest, about
  0.1, is a softer edge.
- `Mynd.Filter.gradient_maxima` keeps the pixels where the strength is a
  maximum along the gradient, and `Mynd.Mask.reconstruct` keeps those
  above the low threshold that join one above the high threshold through
  8-connected pixels. The composition is Canny's edges from
  [`doc/edges.md`](../../doc/edges.md), with `` `Sobel `` in place of
  `` `Central ``, which smooths across each difference as well.
- `Mynd.Mask.components` labels each component by the least index of its
  pixels, so counting the pixels labelled with their own index counts the
  pieces.

It prints the edge pixels and their 8-connected pieces at each pair of
thresholds, and the largest strength. `edges.png` shows the strength and
both edge maps.

Margins: the thinnest decision is `gradient_maxima`'s comparison of a
pixel's strength with the interpolated strengths beside it: among pixels
above 0.01 the closest two differ by a relative `1.9e-6`, 16 to 31
float32 ulps. Every maximum's strength lies at least `7.4e-7` from the
thresholds 0.01, 0.02 and 0.03, over 400 float32 ulps there, and
`2.9e-5` from 0.05. The largest strength lies `3e-4` from a change of
its last printed digit.
