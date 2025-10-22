# Rune Benchmarks

This directory contains benchmarks for the `rune` library. We provide comparative benchmarks against `pytorch`.

## Results Rune Grad

```
┌───────────────────────────────┬──────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼──────────┼─────────┼────────────┤
│ ScalarGrad Medium (Rune)      │  10.24μs │  10.11μs │   3.91kw │   1.00x │       100% │
│ ScalarGrad Large (Rune)       │  10.26μs │  10.21μs │   3.91kw │   1.00x │       100% │
│ ScalarGrad Small (Rune)       │  10.80μs │  10.57μs │   3.91kw │   0.95x │       106% │
│ VectorGrad Small (Rune)       │  27.90μs │  27.75μs │  11.25kw │   0.37x │       273% │
│ VectorGrad Medium (Rune)      │  37.09μs │  37.14μs │  11.25kw │   0.28x │       362% │
│ VectorGrad Large (Rune)       │  48.42μs │  48.35μs │  11.25kw │   0.21x │       473% │
│ HigherOrderGrad Small (Rune)  │ 317.65μs │ 316.83μs │ 114.51kw │   0.03x │      3102% │
│ HigherOrderGrad Medium (Rune) │ 382.69μs │ 382.30μs │ 114.51kw │   0.03x │      3738% │
│ HigherOrderGrad Large (Rune)  │ 478.00μs │ 477.74μs │ 114.51kw │   0.02x │      4668% │
│ MatMulGrad Small (Rune)       │ 857.17μs │   1.35ms │  15.76kw │   0.01x │      8372% │
│ ChainGrad Small (Rune)        │   6.13ms │   8.34ms │ 127.24kw │   0.00x │     59867% │
│ MatMulGrad Medium (Rune)      │  12.96ms │  30.50ms │  15.65kw │   0.00x │    126530% │
│ MatMulGrad Large (Rune)       │  53.55ms │ 179.60ms │  15.65kw │   0.00x │    522964% │
│ ChainGrad Medium (Rune)       │  83.39ms │  88.34ms │ 126.33kw │   0.00x │    814453% │
│ ChainGrad Large (Rune)        │ 257.10ms │ 296.84ms │ 126.33kw │   0.00x │   2510998% │
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
