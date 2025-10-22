# Rune Benchmarks

This directory contains benchmarks for the `rune` library. We provide comparative benchmarks against `pytorch`.

## Results Rune Grad

```
┌───────────────────────────────┬──────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼──────────┼─────────┼────────────┤
│ ScalarGrad Large (Rune)       │  10.44μs │  10.49μs │   3.89kw │   1.00x │       100% │
│ ScalarGrad Small (Rune)       │  10.69μs │  10.62μs │   3.89kw │   0.98x │       102% │
│ ScalarGrad Medium (Rune)      │  10.79μs │  10.80μs │   3.89kw │   0.97x │       103% │
│ VectorGrad Small (Rune)       │  29.43μs │  29.34μs │  11.22kw │   0.35x │       282% │
│ VectorGrad Medium (Rune)      │  38.93μs │  38.78μs │  11.22kw │   0.27x │       373% │
│ VectorGrad Large (Rune)       │  50.38μs │  50.23μs │  11.22kw │   0.21x │       483% │
│ HigherOrderGrad Small (Rune)  │ 340.13μs │ 339.70μs │ 113.84kw │   0.03x │      3259% │
│ HigherOrderGrad Medium (Rune) │ 401.85μs │ 401.02μs │ 113.84kw │   0.03x │      3850% │
│ HigherOrderGrad Large (Rune)  │ 521.17μs │ 519.25μs │ 113.84kw │   0.02x │      4994% │
│ MatMulGrad Small (Rune)       │ 984.13μs │   1.68ms │  15.72kw │   0.01x │      9429% │
│ ChainGrad Small (Rune)        │   6.86ms │   9.65ms │ 126.80kw │   0.00x │     65769% │
│ MatMulGrad Medium (Rune)      │  11.91ms │  28.00ms │  15.62kw │   0.00x │    114067% │
│ MatMulGrad Large (Rune)       │  50.97ms │ 193.28ms │  15.62kw │   0.00x │    488401% │
│ ChainGrad Medium (Rune)       │  74.58ms │  84.07ms │ 125.89kw │   0.00x │    714539% │
│ ChainGrad Large (Rune)        │ 240.34ms │ 286.94ms │ 125.89kw │   0.00x │   2302741% │
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
