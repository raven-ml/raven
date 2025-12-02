# Rune Benchmarks

This directory contains benchmarks for the `rune` library. We provide comparative benchmarks against `pytorch`.

## Results Rune Grad

```
┌───────────────────────────────┬──────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼──────────┼─────────┼────────────┤
│ ScalarGrad Medium (Rune)      │  15.89μs │  15.80μs │   7.56kw │   1.00x │       100% │
│ ScalarGrad Large (Rune)       │  15.90μs │  15.86μs │   7.56kw │   1.00x │       100% │
│ ScalarGrad Small (Rune)       │  16.05μs │  16.04μs │   7.56kw │   0.99x │       101% │
│ VectorGrad Small (Rune)       │  32.40μs │  32.39μs │  14.14kw │   0.49x │       204% │
│ VectorGrad Medium (Rune)      │  38.72μs │  38.62μs │  14.14kw │   0.41x │       244% │
│ VectorGrad Large (Rune)       │  46.97μs │  46.85μs │  14.14kw │   0.34x │       296% │
│ HigherOrderGrad Small (Rune)  │ 315.85μs │ 314.07μs │ 129.23kw │   0.05x │      1988% │
│ HigherOrderGrad Medium (Rune) │ 390.42μs │ 388.73μs │ 129.23kw │   0.04x │      2457% │
│ HigherOrderGrad Large (Rune)  │ 538.70μs │ 537.14μs │ 129.23kw │   0.03x │      3390% │
│ MatMulGrad Small (Rune)       │ 626.49μs │ 889.20μs │  21.82kw │   0.03x │      3942% │
│ ChainGrad Small (Rune)        │   4.22ms │   5.49ms │ 165.53kw │   0.00x │     26572% │
│ MatMulGrad Medium (Rune)      │  10.61ms │  12.35ms │  21.70kw │   0.00x │     66768% │
│ MatMulGrad Large (Rune)       │  36.79ms │  46.00ms │  21.70kw │   0.00x │    231511% │
│ ChainGrad Medium (Rune)       │  77.46ms │  88.48ms │ 164.60kw │   0.00x │    487485% │
│ ChainGrad Large (Rune)        │ 249.58ms │ 299.02ms │ 164.60kw │   0.00x │   1570623% │
└───────────────────────────────┴──────────┴──────────┴──────────┴─────────┴────────────┘
```

## Results PyTorch Grad

```
┌──────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                             │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ ScalarGrad Large (PyTorch)       │  15.76µs │  15.73µs │   6.90w │   1.00x │       100% │
│ ScalarGrad Small (PyTorch)       │  15.91µs │  15.86µs │   6.90w │   0.99x │       101% │
│ ScalarGrad Medium (PyTorch)      │  16.03µs │  15.99µs │   6.90w │   0.98x │       102% │
│ VectorGrad Small (PyTorch)       │  20.25µs │  20.24µs │   8.95w │   0.78x │       128% │
│ VectorGrad Medium (PyTorch)      │  20.60µs │  20.46µs │   8.95w │   0.76x │       131% │
│ VectorGrad Large (PyTorch)       │  21.25µs │  20.96µs │   8.95w │   0.74x │       135% │
│ MatMulGrad Small (PyTorch)       │  37.37µs │  35.78µs │  15.11w │   0.42x │       237% │
│ HigherOrderGrad Small (PyTorch)  │  47.46µs │  47.51µs │  41.39w │   0.33x │       301% │
│ HigherOrderGrad Large (PyTorch)  │  47.55µs │  47.54µs │  38.61w │   0.33x │       302% │
│ HigherOrderGrad Medium (PyTorch) │  48.36µs │  48.30µs │  38.61w │   0.33x │       307% │
│ ChainGrad Small (PyTorch)        │ 189.04µs │ 268.01µs │ 141.66w │   0.08x │      1199% │
│ MatMulGrad Medium (PyTorch)      │ 639.37µs │   1.26ms │ 543.12w │   0.02x │      4057% │
│ ChainGrad Medium (PyTorch)       │ 847.94µs │   2.50ms │  1.32kw │   0.02x │      5380% │
│ ChainGrad Large (PyTorch)        │   3.15ms │  11.92ms │  6.02kw │   0.00x │     20011% │
│ MatMulGrad Large (PyTorch)       │   3.58ms │   7.28ms │  3.18kw │   0.00x │     22742% │
└──────────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```
