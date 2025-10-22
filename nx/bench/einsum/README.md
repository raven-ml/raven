# Nx Einsum Benchmarks

Comparative benchmarks of Nx einsum operations against NumPy.

## Results Nx

```
┌──────────────────────────────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                             │ Time/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────┼──────────┼─────────┼─────────┼────────────┤
│ InnerProduct 50x50 f64 (Nx)      │   4.35μs │  2.87kw │   1.00x │       100% │
│ InnerProduct 100x100 f64 (Nx)    │   4.43μs │  2.87kw │   0.98x │       102% │
│ InnerProduct 200x200 f64 (Nx)    │   4.94μs │  2.87kw │   0.88x │       113% │
│ InnerProduct 50x50 f32 (Nx)      │   5.24μs │  2.87kw │   0.83x │       120% │
│ InnerProduct 100x100 f32 (Nx)    │   5.39μs │  2.87kw │   0.81x │       124% │
│ InnerProduct 200x200 f32 (Nx)    │   5.41μs │  2.87kw │   0.80x │       124% │
│ MatMul 50x50 f32 (Nx)            │  10.79μs │  2.90kw │   0.40x │       248% │
│ MatMul 50x50 f64 (Nx)            │  11.22μs │  2.90kw │   0.39x │       258% │
│ ContractReduce2 50x50 f32 (Nx)   │  12.28μs │  3.74kw │   0.35x │       282% │
│ ContractReduce2 50x50 f64 (Nx)   │  13.69μs │  3.74kw │   0.32x │       314% │
│ ContractReduce1 50x50 f64 (Nx)   │  23.10μs │  4.22kw │   0.19x │       531% │
│ ContractReduce1 50x50 f32 (Nx)   │  25.83μs │  4.22kw │   0.17x │       593% │
│ BatchMatMul 50x50 f32 (Nx)       │  30.14μs │  3.52kw │   0.14x │       692% │
│ BatchMatMul 50x50 f64 (Nx)       │  52.01μs │  3.51kw │   0.08x │      1194% │
│ MatMul 100x100 f32 (Nx)          │ 207.83μs │  2.90kw │   0.02x │      4773% │
│ ContractReduce2 100x100 f32 (Nx) │ 258.34μs │  3.74kw │   0.02x │      5933% │
│ MatMul 100x100 f64 (Nx)          │ 273.68μs │  2.89kw │   0.02x │      6285% │
│ ContractReduce2 100x100 f64 (Nx) │ 277.56μs │  3.73kw │   0.02x │      6374% │
│ ContractReduce1 100x100 f32 (Nx) │ 278.82μs │  4.22kw │   0.02x │      6403% │
│ ContractReduce1 100x100 f64 (Nx) │ 344.83μs │  4.20kw │   0.01x │      7919% │
│ BatchMatMul 100x100 f32 (Nx)     │ 906.04μs │  3.51kw │   0.00x │     20807% │
│ BatchMatMul 100x100 f64 (Nx)     │   1.05ms │  3.51kw │   0.00x │     24216% │
│ ContractReduce2 200x200 f32 (Nx) │   1.89ms │  3.73kw │   0.00x │     43342% │
│ MatMul 200x200 f32 (Nx)          │   1.94ms │  2.89kw │   0.00x │     44474% │
│ ContractReduce2 200x200 f64 (Nx) │   2.52ms │  3.73kw │   0.00x │     57954% │
│ ContractReduce1 200x200 f64 (Nx) │   2.80ms │  4.20kw │   0.00x │     64313% │
│ MatMul 200x200 f64 (Nx)          │   2.81ms │  2.89kw │   0.00x │     64496% │
│ ContractReduce1 200x200 f32 (Nx) │   3.26ms │  4.20kw │   0.00x │     74760% │
│ BatchMatMul 200x200 f32 (Nx)     │   8.78ms │  3.51kw │   0.00x │    201622% │
│ BatchMatMul 200x200 f64 (Nx)     │  11.95ms │  3.51kw │   0.00x │    274388% │
└──────────────────────────────────┴──────────┴─────────┴─────────┴────────────┘
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
