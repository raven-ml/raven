# Nx Einsum Benchmarks

Comparative benchmarks of Nx einsum operations against NumPy.

## Results Nx

```
┌──────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                             │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ InnerProduct 50x50 f64 (Nx)      │   1.90μs │   1.92μs │  1.26kw │   1.00x │       100% │
│ InnerProduct 100x100 f32 (Nx)    │   1.96μs │   1.99μs │  1.26kw │   0.97x │       103% │
│ InnerProduct 100x100 f64 (Nx)    │   1.98μs │   2.00μs │  1.26kw │   0.96x │       104% │
│ InnerProduct 50x50 f32 (Nx)      │   2.06μs │   2.03μs │  1.26kw │   0.92x │       108% │
│ InnerProduct 200x200 f32 (Nx)    │   2.18μs │   2.13μs │  1.26kw │   0.87x │       114% │
│ InnerProduct 512x512 f32 (Nx)    │   2.47μs │   2.44μs │  1.26kw │   0.77x │       130% │
│ InnerProduct 200x200 f64 (Nx)    │   2.56μs │   2.45μs │  1.26kw │   0.74x │       135% │
│ InnerProduct 512x512 f64 (Nx)    │   2.63μs │   2.63μs │  1.26kw │   0.72x │       138% │
│ MatMul 50x50 f32 (Nx)            │   3.22μs │   3.22μs │ 952.00w │   0.59x │       169% │
│ MatMul 50x50 f64 (Nx)            │   4.93μs │   4.95μs │ 952.00w │   0.39x │       259% │
│ MatMul 100x100 f32 (Nx)          │   6.86μs │   6.80μs │ 952.00w │   0.28x │       360% │
│ ContractReduce2 50x50 f64 (Nx)   │   7.29μs │   7.26μs │  4.01kw │   0.26x │       383% │
│ ContractReduce2 50x50 f32 (Nx)   │   7.30μs │   7.31μs │  4.01kw │   0.26x │       383% │
│ ContractReduce1 50x50 f32 (Nx)   │   7.81μs │   7.80μs │  4.01kw │   0.24x │       410% │
│ ContractReduce1 50x50 f64 (Nx)   │   8.01μs │   8.04μs │  4.01kw │   0.24x │       421% │
│ IndependentSum 50x50 f64 (Nx)    │   9.77μs │   9.82μs │  3.81kw │   0.19x │       513% │
│ IndependentSum 50x50 f32 (Nx)    │  10.06μs │  10.09μs │  3.81kw │   0.19x │       528% │
│ BatchMatMul 50x50 f32 (Nx)       │  11.52μs │  11.49μs │  2.91kw │   0.17x │       605% │
│ ContractReduce2 100x100 f32 (Nx) │  14.62μs │  14.62μs │  4.01kw │   0.13x │       768% │
│ ContractReduce2 100x100 f64 (Nx) │  14.96μs │  14.72μs │  4.01kw │   0.13x │       786% │
│ ContractReduce1 100x100 f64 (Nx) │  17.04μs │  17.04μs │  4.01kw │   0.11x │       895% │
│ ContractReduce1 100x100 f32 (Nx) │  17.11μs │  16.96μs │  4.01kw │   0.11x │       899% │
│ MatMul 100x100 f64 (Nx)          │  33.03μs │  33.07μs │ 945.00w │   0.06x │      1734% │
│ BatchMatMul 50x50 f64 (Nx)       │  35.48μs │  35.42μs │  2.91kw │   0.05x │      1864% │
│ ContractReduce2 200x200 f32 (Nx) │  50.95μs │  50.92μs │  4.01kw │   0.04x │      2676% │
│ ContractReduce2 200x200 f64 (Nx) │  53.94μs │  53.27μs │  4.01kw │   0.04x │      2833% │
│ ContractReduce1 200x200 f32 (Nx) │  57.76μs │  57.50μs │  4.01kw │   0.03x │      3033% │
│ BatchMatMul 100x100 f32 (Nx)     │  57.95μs │  57.95μs │  2.91kw │   0.03x │      3044% │
│ ContractReduce1 200x200 f64 (Nx) │  58.40μs │  57.82μs │  4.01kw │   0.03x │      3067% │
│ MatMul 200x200 f32 (Nx)          │  59.92μs │  59.24μs │ 945.00w │   0.03x │      3147% │
│ BatchMatMul 100x100 f64 (Nx)     │ 127.22μs │ 127.21μs │  2.91kw │   0.01x │      6681% │
│ MatMul 200x200 f64 (Nx)          │ 147.15μs │ 145.63μs │ 945.00w │   0.01x │      7728% │
│ IndependentSum 100x100 f64 (Nx)  │ 184.71μs │ 471.85μs │  3.81kw │   0.01x │      9701% │
│ IndependentSum 200x200 f32 (Nx)  │ 185.52μs │ 502.80μs │  3.81kw │   0.01x │      9743% │
│ IndependentSum 200x200 f64 (Nx)  │ 186.57μs │ 508.79μs │  3.81kw │   0.01x │      9799% │
│ IndependentSum 100x100 f32 (Nx)  │ 223.17μs │ 460.18μs │  3.81kw │   0.01x │     11721% │
│ BatchMatMul 200x200 f32 (Nx)     │ 235.95μs │ 234.52μs │  2.91kw │   0.01x │     12392% │
│ IndependentSum 512x512 f64 (Nx)  │ 282.82μs │   1.01ms │  3.81kw │   0.01x │     14853% │
│ IndependentSum 512x512 f32 (Nx)  │ 283.39μs │ 970.86μs │  3.81kw │   0.01x │     14883% │
│ MatMul 512x512 f32 (Nx)          │ 324.12μs │ 424.46μs │ 945.00w │   0.01x │     17022% │
│ ContractReduce2 512x512 f32 (Nx) │ 414.50μs │ 412.71μs │  4.01kw │   0.00x │     21769% │
│ BatchMatMul 200x200 f64 (Nx)     │ 438.50μs │ 437.15μs │  2.91kw │   0.00x │     23029% │
│ ContractReduce2 512x512 f64 (Nx) │ 446.00μs │ 441.93μs │  4.01kw │   0.00x │     23423% │
│ ContractReduce1 512x512 f32 (Nx) │ 448.72μs │ 447.88μs │  4.01kw │   0.00x │     23566% │
│ ContractReduce1 512x512 f64 (Nx) │ 509.76μs │ 507.39μs │  4.01kw │   0.00x │     26772% │
│ MatMul 512x512 f64 (Nx)          │ 766.18μs │   1.20ms │ 945.00w │   0.00x │     40238% │
│ BatchMatMul 512x512 f32 (Nx)     │ 818.47μs │   1.29ms │  2.91kw │   0.00x │     42985% │
│ BatchMatMul 512x512 f64 (Nx)     │   2.48ms │   4.50ms │  2.91kw │   0.00x │    130422% │
└──────────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```

## Results NumPy

```
┌─────────────────────────────────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                                │ Time/Run │ mWd/Run │ Speedup │ vs Fastest │
├─────────────────────────────────────┼──────────┼─────────┼─────────┼────────────┤
│ InnerProduct 100x100 f32 (NumPy)    │   1.14µs │   0.12w │   1.00x │       100% │
│ InnerProduct 50x50 f32 (NumPy)      │   1.14µs │   0.12w │   1.00x │       100% │
│ InnerProduct 50x50 f64 (NumPy)      │   1.14µs │   0.12w │   0.99x │       101% │
│ InnerProduct 100x100 f64 (NumPy)    │   1.16µs │   0.13w │   0.98x │       102% │
│ InnerProduct 200x200 f32 (NumPy)    │   1.16µs │   0.15w │   0.98x │       102% │
│ InnerProduct 200x200 f64 (NumPy)    │   1.23µs │   0.17w │   0.93x │       108% │
│ ContractReduce2 50x50 f32 (NumPy)   │  10.83µs │   0.97w │   0.11x │       952% │
│ ContractReduce2 50x50 f64 (NumPy)   │  15.03µs │   1.29w │   0.08x │      1322% │
│ ContractReduce1 50x50 f32 (NumPy)   │  16.93µs │   1.64w │   0.07x │      1489% │
│ MatMul 50x50 f32 (NumPy)            │  20.34µs │   1.74w │   0.06x │      1789% │
│ MatMul 50x50 f64 (NumPy)            │  26.58µs │   2.77w │   0.04x │      2338% │
│ ContractReduce1 50x50 f64 (NumPy)   │  27.01µs │   2.81w │   0.04x │      2375% │
│ ContractReduce2 100x100 f32 (NumPy) │  57.04µs │   6.33w │   0.02x │      5017% │
│ ContractReduce2 100x100 f64 (NumPy) │  94.44µs │   9.32w │   0.01x │      8306% │
│ MatMul 100x100 f32 (NumPy)          │ 102.29µs │  10.32w │   0.01x │      8996% │
│ ContractReduce1 100x100 f32 (NumPy) │ 104.80µs │  10.60w │   0.01x │      9217% │
│ BatchMatMul 50x50 f32 (NumPy)       │ 108.84µs │  10.09w │   0.01x │      9572% │
│ BatchMatMul 50x50 f64 (NumPy)       │ 134.92µs │  13.38w │   0.01x │     11866% │
│ MatMul 100x100 f64 (NumPy)          │ 161.94µs │  14.91w │   0.01x │     14242% │
│ ContractReduce1 100x100 f64 (NumPy) │ 222.89µs │  24.17w │   0.01x │     19603% │
│ ContractReduce2 200x200 f32 (NumPy) │ 483.26µs │  56.67w │   0.00x │     42501% │
│ BatchMatMul 100x100 f32 (NumPy)     │ 518.96µs │  47.34w │   0.00x │     45641% │
│ MatMul 200x200 f32 (NumPy)          │ 733.17µs │  80.95w │   0.00x │     64480% │
│ BatchMatMul 100x100 f64 (NumPy)     │ 762.60µs │  78.79w │   0.00x │     67068% │
│ ContractReduce2 200x200 f64 (NumPy) │ 843.93µs │ 114.13w │   0.00x │     74221% │
│ ContractReduce1 200x200 f32 (NumPy) │ 937.95µs │ 112.44w │   0.00x │     82490% │
│ MatMul 200x200 f64 (NumPy)          │   1.35ms │ 150.59w │   0.00x │    118410% │
│ ContractReduce1 200x200 f64 (NumPy) │   2.27ms │ 294.93w │   0.00x │    200049% │
│ BatchMatMul 200x200 f32 (NumPy)     │   3.37ms │ 397.35w │   0.00x │    296058% │
│ BatchMatMul 200x200 f64 (NumPy)     │   5.80ms │ 712.76w │   0.00x │    510284% │
└─────────────────────────────────────┴──────────┴─────────┴─────────┴────────────┘
```
