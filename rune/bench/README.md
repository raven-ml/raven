# Rune Benchmarks

This directory contains benchmarks for the `rune` library. We provide comparative benchmarks against `pytorch`.

## Results Rune Grad

```
┌───────────────────────────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                          │ Time/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼─────────┼────────────┤
│ ScalarGrad Medium (Rune)      │  10.12μs │   3.91kw │   1.00x │       100% │
│ ScalarGrad Large (Rune)       │  10.26μs │   3.91kw │   0.99x │       101% │
│ ScalarGrad Small (Rune)       │  10.30μs │   3.91kw │   0.98x │       102% │
│ VectorGrad Small (Rune)       │  26.97μs │  11.25kw │   0.38x │       266% │
│ VectorGrad Medium (Rune)      │  37.06μs │  11.25kw │   0.27x │       366% │
│ VectorGrad Large (Rune)       │  50.05μs │  11.25kw │   0.20x │       495% │
│ HigherOrderGrad Small (Rune)  │ 321.60μs │ 114.51kw │   0.03x │      3178% │
│ HigherOrderGrad Medium (Rune) │ 386.85μs │ 114.51kw │   0.03x │      3822% │
│ HigherOrderGrad Large (Rune)  │ 494.94μs │ 114.51kw │   0.02x │      4890% │
│ MatMulGrad Small (Rune)       │   1.39ms │  15.76kw │   0.01x │     13734% │
│ ChainGrad Small (Rune)        │   8.39ms │ 127.24kw │   0.00x │     82868% │
│ MatMulGrad Medium (Rune)      │  34.04ms │  15.65kw │   0.00x │    336318% │
│ ChainGrad Medium (Rune)       │  88.16ms │ 126.33kw │   0.00x │    871091% │
│ MatMulGrad Large (Rune)       │ 190.19ms │  15.65kw │   0.00x │   1879258% │
│ ChainGrad Large (Rune)        │ 304.90ms │ 126.33kw │   0.00x │   3012602% │
└───────────────────────────────┴──────────┴──────────┴─────────┴────────────┘
```

## Results PyTorch Grad

```
┌──────────────────────────────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                             │ Time/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────┼──────────┼─────────┼─────────┼────────────┤
│ ScalarGrad Large (PyTorch)       │  15.55µs │   6.43w │   1.00x │       100% │
│ ScalarGrad Medium (PyTorch)      │  15.69µs │   6.43w │   0.99x │       101% │
│ ScalarGrad Small (PyTorch)       │  15.91µs │   6.42w │   0.98x │       102% │
│ VectorGrad Small (PyTorch)       │  20.53µs │   8.34w │   0.76x │       132% │
│ VectorGrad Large (PyTorch)       │  20.64µs │   8.34w │   0.75x │       133% │
│ VectorGrad Medium (PyTorch)      │  20.70µs │   8.34w │   0.75x │       133% │
│ MatMulGrad Small (PyTorch)       │  35.11µs │  14.11w │   0.44x │       226% │
│ HigherOrderGrad Medium (PyTorch) │  47.36µs │  36.28w │   0.33x │       304% │
│ HigherOrderGrad Large (PyTorch)  │  47.38µs │  36.28w │   0.33x │       305% │
│ HigherOrderGrad Small (PyTorch)  │  48.26µs │  38.90w │   0.32x │       310% │
│ ChainGrad Small (PyTorch)        │ 257.40µs │ 134.08w │   0.06x │      1655% │
│ MatMulGrad Medium (PyTorch)      │   1.31ms │ 514.08w │   0.01x │      8403% │
│ ChainGrad Medium (PyTorch)       │   2.56ms │  1.25kw │   0.01x │     16450% │
│ MatMulGrad Large (PyTorch)       │   7.09ms │  3.04kw │   0.00x │     45577% │
│ ChainGrad Large (PyTorch)        │  11.19ms │  5.72kw │   0.00x │     71949% │
└──────────────────────────────────┴──────────┴─────────┴─────────┴────────────┘
```
