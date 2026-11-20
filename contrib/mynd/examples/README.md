# Mynd Examples

Each example is a program that builds and runs with the test suite, and
`main.expected` holds what it prints. Read them in order: an image's
values, its colour, resizing it for a model, sampling it at positions,
filters, edges and masks, then the annotations a model's outputs become,
and a derivative through sampling.

Run one from its directory, since it reads `../chelsea.png` and writes its
figure there:

```bash
cd contrib/mynd/examples/08-masks
dune exec ./main.exe
```

| Example | Shows | Key functions |
|---|---|---|
| [`01-values`](./01-values/) | An image's codes and fractions | `Mynd.convert` |
| [`02-colour`](./02-colour/) | Averaging encoded values or light; alpha | `Mynd.srgb_to_linear`, `Mynd.premultiply`, `Colour.luma`, `Colour.luminance` |

Examples that make a picture write it beside `main.ml` with hugin. The
pictures are not committed.

Each printed number is chosen away from the decisions it depends on, such
as a threshold, a rounding of the last digit printed or a tie, so that it
prints the same on every machine. Each example's README states how far.

## chelsea.png

- Chelsea the cat, photographed by Stefan van der Walt.
- Licence: CC0 1.0 by the photographer, as scikit-image's `chelsea()`
  states: "No copyright restrictions. CC0 by the photographer (Stefan van
  der Walt)." scikit-image's `LICENSE.txt` also covers its files under
  BSD-3-Clause, copyright 2009-2022 the scikit-image team.
- Source: scikit-image 0.25.2, `skimage/data/chelsea.png`
  (<https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/chelsea.png>),
  SHA-256 `596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb`.
- Changed: its `iCCP`, `iTXt` and `pHYs` chunks are removed and an `sRGB`
  chunk is added; its `IHDR` and `IDAT` chunks, and so its pixels, are
  unchanged. 451 × 300, 8-bit RGB, sRGB values, SHA-256
  `1d2c1e38abd6edaf13b8294724bd3f93fb4f92bc20d43b502382a6b5ac0724c0`.
