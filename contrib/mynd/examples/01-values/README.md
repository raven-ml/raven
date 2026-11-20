# `01-values`

Load the photograph and read its numbers as codes and as fractions of full
scale.

```bash
dune exec ./main.exe
```

- `Nx_io.load_image` gives the file's 8-bit codes, a uint8 array
  `[300; 451; 3]`.
- `Mynd.convert` changes the dtype and keeps each value's fraction: code
  `k` of uint8 is `k / 255` in float32 and `257 k` in uint16 (the Values
  paragraph of [`mynd.mli`](../../lib/mynd.mli)).
- Converting back to uint8 rounds to the nearest code, so every code comes
  back unchanged.
- `Nx.cast` truncates. After a gamma of 0.8, truncating `255 x` gives one
  code less than `convert` wherever the fraction of `255 x` is at least a
  half.

It prints the shape, one pixel in the three dtypes, the codes a round trip
changes and how many codes truncation lowers.

Margins: the pixel's float32 values lie at least `1.5e-5` from a change
of their fourth decimal. Over the 256 codes, the fraction of `255 x^0.8`
lies at least `1.4e-3` from a half and `6.2e-4` from a whole number,
where float32 rounds within `3e-5`.
