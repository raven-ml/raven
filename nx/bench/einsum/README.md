# Nx Einsum Benchmarks

Comparative benchmarks of Nx einsum operations against NumPy.

## Results Nx

```
┌──────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                             │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ InnerProduct 100x100 f64 (Nx)    │   3.57μs │   3.51μs │ 973.00w │   1.00x │       100% │
│ InnerProduct 100x100 f32 (Nx)    │   3.76μs │   3.79μs │ 973.00w │   0.95x │       105% │
│ InnerProduct 50x50 f64 (Nx)      │   3.93μs │   3.73μs │ 973.00w │   0.91x │       110% │
│ InnerProduct 50x50 f32 (Nx)      │   3.96μs │   3.91μs │ 973.00w │   0.90x │       111% │
│ InnerProduct 200x200 f64 (Nx)    │   4.18μs │   4.08μs │ 973.00w │   0.85x │       117% │
│ InnerProduct 200x200 f32 (Nx)    │   4.60μs │   4.31μs │ 973.00w │   0.78x │       129% │
│ MatMul 50x50 f32 (Nx)            │   4.84μs │   4.86μs │ 759.00w │   0.74x │       135% │
│ MatMul 50x50 f64 (Nx)            │   7.37μs │   7.37μs │ 759.00w │   0.48x │       206% │
│ MatMul 100x100 f32 (Nx)          │  10.76μs │  10.65μs │ 759.00w │   0.33x │       301% │
│ ContractReduce2 50x50 f32 (Nx)   │  16.79μs │  16.56μs │  3.72kw │   0.21x │       470% │
│ ContractReduce2 50x50 f64 (Nx)   │  19.57μs │  19.62μs │  3.72kw │   0.18x │       548% │
│ BatchMatMul 50x50 f32 (Nx)       │  20.00μs │  19.90μs │  3.28kw │   0.18x │       560% │
│ ContractReduce1 50x50 f64 (Nx)   │  36.42μs │  36.28μs │  4.05kw │   0.10x │      1020% │
│ ContractReduce1 50x50 f32 (Nx)   │  41.74μs │  41.51μs │  4.05kw │   0.09x │      1169% │
│ MatMul 100x100 f64 (Nx)          │  53.92μs │  53.05μs │ 752.00w │   0.07x │      1510% │
│ BatchMatMul 50x50 f64 (Nx)       │  66.27μs │  65.17μs │  3.27kw │   0.05x │      1856% │
│ MatMul 200x200 f32 (Nx)          │  97.01μs │  96.41μs │ 752.00w │   0.04x │      2716% │
│ BatchMatMul 100x100 f32 (Nx)     │ 115.59μs │ 110.53μs │  3.27kw │   0.03x │      3237% │
│ ContractReduce2 100x100 f32 (Nx) │ 151.21μs │ 354.03μs │  3.72kw │   0.02x │      4234% │
│ ContractReduce2 100x100 f64 (Nx) │ 202.63μs │ 407.80μs │  3.71kw │   0.02x │      5674% │
│ BatchMatMul 100x100 f64 (Nx)     │ 216.94μs │ 212.00μs │  3.27kw │   0.02x │      6075% │
│ MatMul 200x200 f64 (Nx)          │ 246.41μs │ 242.44μs │ 752.00w │   0.01x │      6900% │
│ ContractReduce1 100x100 f32 (Nx) │ 261.97μs │ 480.96μs │  4.05kw │   0.01x │      7336% │
│ ContractReduce2 200x200 f32 (Nx) │ 318.78μs │ 614.09μs │  3.71kw │   0.01x │      8927% │
│ ContractReduce1 100x100 f64 (Nx) │ 321.52μs │ 543.43μs │  4.04kw │   0.01x │      9003% │
│ BatchMatMul 200x200 f32 (Nx)     │ 435.39μs │ 352.82μs │  3.27kw │   0.01x │     12192% │
│ ContractReduce2 200x200 f64 (Nx) │ 544.39μs │ 801.98μs │  3.71kw │   0.01x │     15244% │
│ BatchMatMul 200x200 f64 (Nx)     │ 735.71μs │ 682.68μs │  3.27kw │   0.00x │     20602% │
│ ContractReduce1 200x200 f32 (Nx) │ 855.36μs │   1.08ms │  4.04kw │   0.00x │     23953% │
│ ContractReduce1 200x200 f64 (Nx) │   1.08ms │   1.37ms │  4.04kw │   0.00x │     30147% │
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
