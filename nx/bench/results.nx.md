# Nx Benchmarks

## Native

┌─────────────────────────────────────┬──────────┬──────────┬─────────────┐
│ Name                                │ Time/Run │  mWd/Run │  vs Fastest │
├─────────────────────────────────────┼──────────┼──────────┼─────────────┤
│ Mul 50x50 float32 (Native)          │  50.61μs │  190.08w │     100.00% │
│ Square 50x50 float32 (Native)       │  50.67μs │  190.08w │     100.13% │
│ Add 50x50 float32 (Native)          │  53.46μs │  190.08w │     105.63% │
│ Sum 50x50 float32 (Native)          │  58.09μs │   1.51kw │     114.78% │
│ Mul 100x100 float32 (Native)        │  63.70μs │  190.08w │     125.87% │
│ Add 100x100 float32 (Native)        │  64.01μs │  190.08w │     126.48% │
│ Square 100x100 float32 (Native)     │  64.29μs │  190.08w │     127.03% │
│ Sum 100x100 float32 (Native)        │  68.82μs │   4.51kw │     135.98% │
│ MatMul 50x50 float32 (Native)       │  78.16μs │  212.13w │     154.44% │
│ Sum 200x200 float32 (Native)        │ 112.22μs │  16.51kw │     221.76% │
│ Add 200x200 float32 (Native)        │ 127.09μs │  183.24w │     251.14% │
│ Square 200x200 float32 (Native)     │ 136.64μs │  183.24w │     270.00% │
│ Mul 200x200 float32 (Native)        │ 140.42μs │  183.24w │     277.47% │
│ MatMul 100x100 float32 (Native)     │ 235.07μs │  228.24w │     464.50% │
│ Add 500x500 float32 (Native)        │ 258.65μs │  183.37w │     511.09% │
│ Mul 500x500 float32 (Native)        │ 273.31μs │  183.37w │     540.05% │
│ Square 500x500 float32 (Native)     │ 275.41μs │  183.37w │     544.20% │
│ Sum 500x500 float32 (Native)        │ 442.73μs │ 100.51kw │     874.83% │
│ MatMul 200x200 float32 (Native)     │   1.54ms │  332.62w │    3051.87% │
│ Conv2D 3x3 50x50 float32 (Native)   │  13.79ms │   1.02Mw │   27243.40% │
│ Conv2D 5x5 50x50 float32 (Native)   │  33.18ms │   2.58Mw │   65560.04% │
│ Conv2D 3x3 100x100 float32 (Native) │  55.92ms │   4.23Mw │  110496.58% │
│ Conv2D 5x5 100x100 float32 (Native) │ 166.29ms │  11.16Mw │  328597.20% │
│ Conv2D 3x3 200x200 float32 (Native) │ 227.61ms │  17.26Mw │  449748.55% │
│ Conv2D 5x5 200x200 float32 (Native) │ 554.60ms │  46.43Mw │ 1095878.49% │
└─────────────────────────────────────┴──────────┴──────────┴─────────────┘

## CBLAS

┌────────────────────────────────────┬──────────┬─────────┬─────────────┐
│ Name                               │ Time/Run │ mWd/Run │  vs Fastest │
├────────────────────────────────────┼──────────┼─────────┼─────────────┤
│ Add 50x50 float32 (CBLAS)          │   1.03μs │  43.76w │     100.00% │
│ Mul 50x50 float32 (CBLAS)          │   1.04μs │  43.79w │     100.96% │
│ Square 50x50 float32 (CBLAS)       │   1.10μs │  43.79w │     106.61% │
│ MatMul 50x50 float32 (CBLAS)       │   2.10μs │  40.81w │     203.97% │
│ Sum 50x50 float32 (CBLAS)          │   3.74μs │  61.84w │     363.39% │
│ Square 100x100 float32 (CBLAS)     │   4.46μs │  43.87w │     433.26% │
│ MatMul 100x100 float32 (CBLAS)     │   7.70μs │  40.90w │     747.93% │
│ Mul 100x100 float32 (CBLAS)        │   8.84μs │  43.93w │     858.57% │
│ Add 100x100 float32 (CBLAS)        │   8.85μs │  43.96w │     858.73% │
│ Sum 100x100 float32 (CBLAS)        │  13.50μs │  61.96w │    1310.40% │
│ Add 200x200 float32 (CBLAS)        │  44.49μs │  37.13w │    4319.17% │
│ MatMul 200x200 float32 (CBLAS)     │  49.27μs │  34.13w │    4783.18% │
│ Sum 200x200 float32 (CBLAS)        │  52.57μs │  62.08w │    5104.09% │
│ Square 200x200 float32 (CBLAS)     │  80.45μs │  37.18w │    7810.97% │
│ Mul 200x200 float32 (CBLAS)        │  83.09μs │  37.18w │    8066.83% │
│ Square 500x500 float32 (CBLAS)     │ 128.38μs │  37.24w │   12463.28% │
│ Mul 500x500 float32 (CBLAS)        │ 131.97μs │  37.24w │   12812.48% │
│ Add 500x500 float32 (CBLAS)        │ 138.46μs │  37.24w │   13442.41% │
│ Sum 500x500 float32 (CBLAS)        │ 325.03μs │  62.30w │   31555.88% │
│ Conv2D 3x3 50x50 float32 (CBLAS)   │ 340.16μs │ 378.37w │   33024.25% │
│ Conv2D 5x5 50x50 float32 (CBLAS)   │ 710.88μs │ 378.53w │   69015.92% │
│ Conv2D 3x3 100x100 float32 (CBLAS) │   1.20ms │ 378.62w │  116016.75% │
│ Conv2D 5x5 100x100 float32 (CBLAS) │   3.01ms │ 378.85w │  292335.19% │
│ Conv2D 3x3 200x200 float32 (CBLAS) │   4.67ms │ 378.99w │  453061.48% │
│ Conv2D 5x5 200x200 float32 (CBLAS) │  12.40ms │ 379.55w │ 1204067.72% │
└────────────────────────────────────┴──────────┴─────────┴─────────────┘

## Metal

┌────────────────────────────────────┬──────────┬─────────┬────────────┐
│ Name                               │ Time/Run │ mWd/Run │ vs Fastest │
├────────────────────────────────────┼──────────┼─────────┼────────────┤
│ Mul 50x50 float32 (Metal)          │ 305.64μs │ 10.82kw │    100.00% │
│ Add 50x50 float32 (Metal)          │ 306.35μs │ 10.82kw │    100.23% │
│ Square 50x50 float32 (Metal)       │ 308.91μs │ 10.82kw │    101.07% │
│ MatMul 50x50 float32 (Metal)       │ 311.74μs │ 10.80kw │    102.00% │
│ Square 100x100 float32 (Metal)     │ 320.09μs │ 10.82kw │    104.73% │
│ Add 100x100 float32 (Metal)        │ 325.36μs │ 10.82kw │    106.45% │
│ Mul 100x100 float32 (Metal)        │ 332.06μs │ 10.82kw │    108.64% │
│ MatMul 100x100 float32 (Metal)     │ 334.62μs │ 10.80kw │    109.48% │
│ Add 200x200 float32 (Metal)        │ 339.95μs │ 10.82kw │    111.23% │
│ Square 200x200 float32 (Metal)     │ 346.61μs │ 10.82kw │    113.40% │
│ Mul 200x200 float32 (Metal)        │ 352.60μs │ 10.82kw │    115.36% │
│ Sum 50x50 float32 (Metal)          │ 360.13μs │ 12.85kw │    117.83% │
│ MatMul 200x200 float32 (Metal)     │ 373.88μs │ 10.80kw │    122.32% │
│ Add 500x500 float32 (Metal)        │ 483.83μs │ 10.82kw │    158.30% │
│ Sum 100x100 float32 (Metal)        │ 487.65μs │ 12.85kw │    159.55% │
│ Mul 500x500 float32 (Metal)        │ 498.67μs │ 10.82kw │    163.15% │
│ Square 500x500 float32 (Metal)     │ 541.25μs │ 10.82kw │    177.09% │
│ Sum 200x200 float32 (Metal)        │ 812.29μs │ 12.85kw │    265.77% │
│ Conv2D 3x3 50x50 float32 (Metal)   │   1.02ms │ 33.71kw │    335.07% │
│ Conv2D 5x5 50x50 float32 (Metal)   │   1.15ms │ 33.71kw │    377.20% │
│ Conv2D 3x3 100x100 float32 (Metal) │   1.52ms │ 33.71kw │    498.15% │
│ Conv2D 5x5 100x100 float32 (Metal) │   1.73ms │ 33.71kw │    566.65% │
│ Conv2D 3x3 200x200 float32 (Metal) │   1.90ms │ 33.71kw │    621.44% │
│ Sum 500x500 float32 (Metal)        │   2.02ms │ 12.85kw │    661.62% │
│ Conv2D 5x5 200x200 float32 (Metal) │   2.85ms │ 33.71kw │    932.67% │
└────────────────────────────────────┴──────────┴─────────┴────────────┘
