# Nx Einsum Benchmarks

Comparative benchmarks of Nx einsum operations against NumPy.

## Results Nx

```
┌───────────────────────────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                          │ Time/Run │ mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼─────────┼─────────┼────────────┤
│ InnerProduct 50x50 f64 (Nx)   │   4.05μs │  2.87kw │   1.00x │       100% │
│ InnerProduct 100x100 f64 (Nx) │   4.48μs │  2.87kw │   0.91x │       110% │
│ InnerProduct 200x200 f64 (Nx) │   5.01μs │  2.87kw │   0.81x │       123% │
│ InnerProduct 50x50 f32 (Nx)   │   5.17μs │  2.87kw │   0.78x │       128% │
│ InnerProduct 100x100 f32 (Nx) │   5.43μs │  2.87kw │   0.75x │       134% │
│ InnerProduct 200x200 f32 (Nx) │   5.56μs │  2.87kw │   0.73x │       137% │
│ MatMul 50x50 f32 (Nx)         │   9.98μs │  2.90kw │   0.41x │       246% │
│ MatMul 50x50 f64 (Nx)         │  11.62μs │  2.90kw │   0.35x │       287% │
│ BatchMatMul 50x50 f32 (Nx)    │  29.73μs │  3.52kw │   0.14x │       733% │
│ BatchMatMul 50x50 f64 (Nx)    │  52.78μs │  3.51kw │   0.08x │      1302% │
│ MatMul 100x100 f32 (Nx)       │ 242.45μs │  2.90kw │   0.02x │      5980% │
│ MatMul 100x100 f64 (Nx)       │ 270.58μs │  2.89kw │   0.01x │      6674% │
│ BatchMatMul 100x100 f32 (Nx)  │ 870.03μs │  3.51kw │   0.00x │     21459% │
│ BatchMatMul 100x100 f64 (Nx)  │   1.08ms │  3.51kw │   0.00x │     26698% │
│ MatMul 200x200 f64 (Nx)       │   4.26ms │  2.89kw │   0.00x │    105084% │
│ MatMul 200x200 f32 (Nx)       │   4.56ms │  2.89kw │   0.00x │    112457% │
│ BatchMatMul 200x200 f32 (Nx)  │  12.73ms │  3.51kw │   0.00x │    313876% │
│ BatchMatMul 200x200 f64 (Nx)  │  29.70ms │  3.51kw │   0.00x │    732604% │
└───────────────────────────────┴──────────┴─────────┴─────────┴────────────┘
```

## Results NumPy

```
┌──────────────────────────────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                             │ Time/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────┼──────────┼─────────┼─────────┼────────────┤
│ InnerProduct 200x200 f32 (NumPy) │   1.18µs │   0.15w │   1.00x │       100% │
│ InnerProduct 200x200 f64 (NumPy) │   1.24µs │   0.17w │   0.95x │       105% │
│ InnerProduct 50x50 f64 (NumPy)   │   1.27µs │   0.13w │   0.94x │       107% │
│ InnerProduct 100x100 f32 (NumPy) │   1.27µs │   0.14w │   0.93x │       107% │
│ InnerProduct 50x50 f32 (NumPy)   │   1.27µs │   0.13w │   0.93x │       107% │
│ InnerProduct 100x100 f64 (NumPy) │   1.28µs │   0.14w │   0.93x │       108% │
│ MatMul 50x50 f32 (NumPy)         │  21.49µs │   2.36w │   0.06x │      1814% │
│ MatMul 50x50 f64 (NumPy)         │  29.37µs │   3.09w │   0.04x │      2480% │
│ BatchMatMul 50x50 f32 (NumPy)    │ 116.47µs │  11.26w │   0.01x │      9835% │
│ MatMul 100x100 f32 (NumPy)       │ 118.16µs │  11.36w │   0.01x │      9979% │
│ BatchMatMul 50x50 f64 (NumPy)    │ 153.45µs │  14.79w │   0.01x │     12958% │
│ MatMul 100x100 f64 (NumPy)       │ 194.95µs │  19.49w │   0.01x │     16463% │
│ BatchMatMul 100x100 f32 (NumPy)  │ 586.20µs │  64.29w │   0.00x │     49503% │
│ MatMul 200x200 f32 (NumPy)       │ 746.27µs │  86.03w │   0.00x │     63020% │
│ BatchMatMul 100x100 f64 (NumPy)  │ 860.42µs │  91.76w │   0.00x │     72660% │
│ MatMul 200x200 f64 (NumPy)       │   1.36ms │ 152.36w │   0.00x │    114924% │
│ BatchMatMul 200x200 f32 (NumPy)  │   3.43ms │ 416.19w │   0.00x │    289512% │
│ BatchMatMul 200x200 f64 (NumPy)  │   5.89ms │ 705.33w │   0.00x │    497470% │
└──────────────────────────────────┴──────────┴─────────┴─────────┴────────────┘
```
