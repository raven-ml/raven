# Rune Benchmarks

This directory contains benchmarks for the `rune` library. We provide comparative benchmarks against `pytorch`.

## Results Rune Grad

```
┌───────────────────────────────┬──────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼──────────┼─────────┼────────────┤
│ ScalarGrad Large (Rune)       │   9.10μs │   9.06μs │   3.46kw │   1.00x │       100% │
│ ScalarGrad Medium (Rune)      │   9.57μs │   9.43μs │   3.46kw │   0.95x │       105% │
│ ScalarGrad Small (Rune)       │   9.84μs │   9.51μs │   3.46kw │   0.92x │       108% │
│ VectorGrad Small (Rune)       │  24.58μs │  24.37μs │   9.50kw │   0.37x │       270% │
│ VectorGrad Medium (Rune)      │  31.24μs │  30.96μs │   9.50kw │   0.29x │       343% │
│ VectorGrad Large (Rune)       │  39.15μs │  38.90μs │   9.50kw │   0.23x │       430% │
│ HigherOrderGrad Small (Rune)  │ 261.77μs │ 258.28μs │  90.93kw │   0.03x │      2878% │
│ HigherOrderGrad Medium (Rune) │ 307.21μs │ 306.49μs │  90.93kw │   0.03x │      3378% │
│ HigherOrderGrad Large (Rune)  │ 378.72μs │ 378.57μs │  90.93kw │   0.02x │      4164% │
│ MatMulGrad Small (Rune)       │ 570.87μs │ 841.82μs │  13.41kw │   0.02x │      6276% │
│ ChainGrad Small (Rune)        │   3.89ms │   5.29ms │ 114.91kw │   0.00x │     42790% │
│ MatMulGrad Medium (Rune)      │   8.70ms │  10.32ms │  13.31kw │   0.00x │     95675% │
│ MatMulGrad Large (Rune)       │  29.81ms │  38.36ms │  13.31kw │   0.00x │    327767% │
│ ChainGrad Medium (Rune)       │  65.81ms │  76.24ms │ 114.13kw │   0.00x │    723543% │
│ ChainGrad Large (Rune)        │ 219.58ms │ 262.43ms │ 114.13kw │   0.00x │   2414100% │
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
