# Nx Einsum Benchmarks

Comparative benchmarks of Nx einsum operations against NumPy.

## Results Nx

```
┌──────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                             │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ InnerProduct 50x50 f32 (Nx)      │   1.87μs │   1.94μs │  1.28kw │   1.00x │       100% │
│ InnerProduct 50x50 f64 (Nx)      │   1.92μs │   1.92μs │  1.28kw │   0.98x │       102% │
│ InnerProduct 100x100 f32 (Nx)    │   1.99μs │   1.95μs │  1.28kw │   0.94x │       106% │
│ InnerProduct 200x200 f64 (Nx)    │   2.11μs │   2.13μs │  1.28kw │   0.89x │       113% │
│ InnerProduct 200x200 f32 (Nx)    │   2.14μs │   2.13μs │  1.28kw │   0.87x │       114% │
│ InnerProduct 512x512 f32 (Nx)    │   2.63μs │   2.64μs │  1.28kw │   0.71x │       140% │
│ InnerProduct 512x512 f64 (Nx)    │   2.65μs │   2.65μs │  1.28kw │   0.71x │       141% │
│ InnerProduct 100x100 f64 (Nx)    │   2.66μs │   2.67μs │  1.28kw │   0.70x │       142% │
│ MatMul 50x50 f32 (Nx)            │   3.32μs │   3.35μs │ 952.00w │   0.56x │       177% │
│ MatMul 50x50 f64 (Nx)            │   4.93μs │   4.94μs │ 952.00w │   0.38x │       263% │
│ MatMul 100x100 f32 (Nx)          │   7.20μs │   7.12μs │ 952.00w │   0.26x │       385% │
│ ContractReduce2 50x50 f64 (Nx)   │   7.91μs │   7.92μs │  4.60kw │   0.24x │       423% │
│ ContractReduce2 50x50 f32 (Nx)   │   8.16μs │   8.14μs │  4.60kw │   0.23x │       436% │
│ ContractReduce1 50x50 f64 (Nx)   │   8.35μs │   8.38μs │  4.60kw │   0.22x │       446% │
│ ContractReduce1 50x50 f32 (Nx)   │   9.22μs │   8.56μs │  4.60kw │   0.20x │       492% │
│ IndependentSum 50x50 f32 (Nx)    │  10.11μs │  10.04μs │  4.26kw │   0.19x │       540% │
│ IndependentSum 50x50 f64 (Nx)    │  10.22μs │  10.17μs │  4.26kw │   0.18x │       546% │
│ BatchMatMul 50x50 f32 (Nx)       │  13.24μs │  13.15μs │  3.74kw │   0.14x │       707% │
│ ContractReduce2 100x100 f64 (Nx) │  15.37μs │  15.32μs │  4.60kw │   0.12x │       821% │
│ ContractReduce2 100x100 f32 (Nx) │  15.73μs │  15.67μs │  4.60kw │   0.12x │       840% │
│ ContractReduce1 100x100 f64 (Nx) │  17.53μs │  17.53μs │  4.60kw │   0.11x │       936% │
│ ContractReduce1 100x100 f32 (Nx) │  17.56μs │  17.50μs │  4.60kw │   0.11x │       938% │
│ MatMul 100x100 f64 (Nx)          │  31.89μs │  31.94μs │ 945.00w │   0.06x │      1704% │
│ BatchMatMul 50x50 f64 (Nx)       │  36.80μs │  36.84μs │  3.73kw │   0.05x │      1966% │
│ ContractReduce2 200x200 f32 (Nx) │  52.48μs │  52.42μs │  4.60kw │   0.04x │      2803% │
│ ContractReduce2 200x200 f64 (Nx) │  56.38μs │  53.42μs │  4.60kw │   0.03x │      3012% │
│ MatMul 200x200 f32 (Nx)          │  57.39μs │  57.24μs │ 945.00w │   0.03x │      3065% │
│ ContractReduce1 200x200 f32 (Nx) │  57.44μs │  57.28μs │  4.60kw │   0.03x │      3068% │
│ BatchMatMul 100x100 f32 (Nx)     │  59.04μs │  58.26μs │  3.73kw │   0.03x │      3154% │
│ ContractReduce1 200x200 f64 (Nx) │  59.69μs │  58.85μs │  4.60kw │   0.03x │      3188% │
│ BatchMatMul 100x100 f64 (Nx)     │ 128.10μs │ 127.97μs │  3.73kw │   0.01x │      6842% │
│ MatMul 200x200 f64 (Nx)          │ 138.63μs │ 138.64μs │ 945.00w │   0.01x │      7404% │
│ IndependentSum 100x100 f32 (Nx)  │ 181.36μs │ 454.97μs │  4.26kw │   0.01x │      9687% │
│ IndependentSum 200x200 f64 (Nx)  │ 184.16μs │ 503.79μs │  4.26kw │   0.01x │      9836% │
│ IndependentSum 100x100 f64 (Nx)  │ 184.56μs │ 475.76μs │  4.26kw │   0.01x │      9858% │
│ IndependentSum 200x200 f32 (Nx)  │ 193.35μs │ 516.52μs │  4.26kw │   0.01x │     10327% │
│ BatchMatMul 200x200 f32 (Nx)     │ 220.81μs │ 220.08μs │  3.73kw │   0.01x │     11794% │
│ IndependentSum 512x512 f32 (Nx)  │ 280.05μs │ 965.37μs │  4.26kw │   0.01x │     14958% │
│ IndependentSum 512x512 f64 (Nx)  │ 280.64μs │ 971.81μs │  4.26kw │   0.01x │     14989% │
│ MatMul 512x512 f32 (Nx)          │ 329.47μs │ 429.62μs │ 945.00w │   0.01x │     17598% │
│ ContractReduce2 512x512 f32 (Nx) │ 409.10μs │ 408.72μs │  4.60kw │   0.00x │     21851% │
│ ContractReduce2 512x512 f64 (Nx) │ 427.18μs │ 426.76μs │  4.60kw │   0.00x │     22816% │
│ BatchMatMul 200x200 f64 (Nx)     │ 430.97μs │ 430.47μs │  3.73kw │   0.00x │     23019% │
│ ContractReduce1 512x512 f32 (Nx) │ 460.26μs │ 455.87μs │  4.60kw │   0.00x │     24584% │
│ ContractReduce1 512x512 f64 (Nx) │ 513.63μs │ 511.38μs │  4.60kw │   0.00x │     27434% │
│ MatMul 512x512 f64 (Nx)          │ 708.86μs │   1.13ms │ 945.00w │   0.00x │     37862% │
│ BatchMatMul 512x512 f32 (Nx)     │ 838.08μs │   1.33ms │  3.73kw │   0.00x │     44764% │
│ BatchMatMul 512x512 f64 (Nx)     │   2.48ms │   4.49ms │  3.73kw │   0.00x │    132590% │
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
