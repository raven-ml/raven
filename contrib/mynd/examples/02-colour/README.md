# `02-colour`

An image holds the values its caller gives it, sRGB-encoded for a
photograph, and every function computes on them as they are. This example
averages the same pixels in encoded values and in light, and colours with
alpha straight and premultiplied.

```bash
dune exec ./main.exe
```

- `Mynd.Sample.resize` averages a checkerboard of black and white into
  0.5 in encoded values. Through `Mynd.srgb_to_linear` and
  `Mynd.linear_to_srgb`, it averages light: 0.5 in light, which sRGB
  encodes as 0.735 (the Encoding paragraph of
  [`mynd.mli`](../../lib/mynd.mli)).
- The photograph quartered both ways shows how far the two differ on a
  real image: under a code on average, more where light and dark meet.
- `Mynd.premultiply` before averaging and `Mynd.unpremultiply` after keep
  a transparent pixel's colour out of the mean.
- `Mynd.Colour.luma` weighs encoded values and `Mynd.Colour.luminance`
  weighs light, so the same green reads 0.587 and 0.715.

`colour.png` shows the photograph quartered both ways.

Margins: the checkerboard's encoded mean and the pair's colour and alpha
means are exact, and a value that rounds to zero prints without a sign.
The light average lies `1.4e-4` from a change of its last printed digit,
the luma `5e-4` and the luminance `3.3e-4`. The photograph's means lie at
least `0.02` codes from a change of their last digit, the mean difference
`2.2e-3` codes and the largest `0.015` codes.
