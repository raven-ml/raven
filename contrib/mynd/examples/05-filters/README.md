# `05-filters`

Blur salt-and-pepper noise out of the grey photograph three ways, and see
what a blur reads past the border.

```bash
dune exec ./main.exe
```

- `Mynd.Colour.luma` gives the grey photograph; one pixel in ten is set
  to black or white, drawn from a seeded `Random.State`.
- A box blur is two `Mynd.Filter.correlate` calls, by a column and by a
  row of `1 / 5`, as [`doc/filters.md`](../../doc/filters.md) builds it
  (Box blur).
- `Mynd.Filter.gaussian` weighs the window by a Gaussian of deviation 1,
  and `Mynd.Filter.median` takes its median. The median ignores the
  noise's extreme values, so its error is the lowest.
- Every filter takes an edge rule, which says what it reads past the
  border (the section on reading outside the image in
  [`mynd.mli`](../../lib/mynd.mli)). Under `` `Symmetric_whole `` it
  reads the image mirrored about its border pixels; under `` `Zero `` it
  reads black, and the blurred top row darkens.

It prints each result's mean error to the photograph in codes of 255, and
the mean code of the top row before and after blurring under each rule.
`filters.png` shows the noisy, Gaussian and median images.

Margins: the mean errors lie at least `6e-4` codes from a change of
their last digit, and the top row's means at least `0.037` codes.
