# Rune Performance Benchmarks

Comprehensive benchmark results for Rune tensor operations across forward pass, backward pass (automatic differentiation), and JIT compilation modes.

## Benchmark Results
┌──────────────────────────────────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                                 │ Time/Run │  mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────────┼──────────┼──────────┼─────────┼────────────┤
│ Transpose 200x200 f32 (forward)      │ 564.51ns │  202.00w │   1.00x │       100% │
│ Transpose 50x50 f32 (forward)        │ 600.38ns │  202.00w │   0.94x │       106% │
│ Transpose 100x100 f32 (forward)      │ 632.75ns │  202.00w │   0.89x │       112% │
│ Reshape 200x200 f32 (forward)        │ 660.90ns │  288.00w │   0.85x │       117% │
│ Reshape 100x100 f32 (forward)        │ 710.70ns │  288.00w │   0.79x │       126% │
│ Reshape 50x50 f32 (forward)          │ 711.32ns │  288.00w │   0.79x │       126% │
│ Transpose 100x100 f32 (jit)          │   1.14μs │  399.00w │   0.50x │       202% │
│ Transpose 200x200 f32 (jit)          │   1.14μs │  399.00w │   0.50x │       202% │
│ Reshape 200x200 f32 (jit)            │   1.37μs │  485.00w │   0.41x │       242% │
│ Reshape 50x50 f32 (jit)              │   1.39μs │  485.00w │   0.41x │       246% │
│ Transpose 50x50 f32 (jit)            │   1.50μs │  399.00w │   0.38x │       265% │
│ Slice 100x100 f32 (forward)          │   1.50μs │  702.00w │   0.38x │       265% │
│ Slice 200x200 f32 (forward)          │   1.58μs │  702.00w │   0.36x │       280% │
│ Reshape 100x100 f32 (jit)            │   1.68μs │  485.00w │   0.34x │       298% │
│ Slice 50x50 f32 (forward)            │   1.81μs │  702.00w │   0.31x │       320% │
│ Matmul 50x50 f32 (forward)           │   7.02μs │   1.72kw │   0.08x │      1243% │
│ Matmul 50x50 f32 (jit)               │   7.32μs │   1.91kw │   0.08x │      1297% │
│ Add 100x100 f32 (forward)            │  12.73μs │   1.84kw │   0.04x │      2255% │
│ Max 50x50 f32 (forward)              │  13.74μs │  940.00w │   0.04x │      2434% │
│ Sum 50x50 f32 (forward)              │  15.22μs │  940.00w │   0.04x │      2696% │
│ Add 100x100 f32 (jit)                │  15.26μs │   2.04kw │   0.04x │      2704% │
│ Sum 50x50 f32 (jit)                  │  15.68μs │   1.14kw │   0.04x │      2777% │
│ Square 100x100 f32 (forward)         │  15.75μs │   1.84kw │   0.04x │      2790% │
│ Mean 50x50 f32 (forward)             │  15.95μs │   1.44kw │   0.04x │      2826% │
│ Add 50x50 f32 (forward)              │  16.60μs │   1.84kw │   0.03x │      2940% │
│ BatchedMatmul 50x50 f32 (jit)        │  16.71μs │   2.13kw │   0.03x │      2960% │
│ BatchedMatmul 50x50 f32 (forward)    │  17.30μs │   1.93kw │   0.03x │      3064% │
│ Mul 100x100 f32 (forward)            │  17.68μs │   1.84kw │   0.03x │      3132% │
│ Sin 50x50 f32 (forward)              │  18.10μs │  945.00w │   0.03x │      3206% │
│ Mul 50x50 f32 (forward)              │  18.36μs │   1.84kw │   0.03x │      3252% │
│ Mean 50x50 f32 (jit)                 │  18.92μs │   1.63kw │   0.03x │      3351% │
│ Square 50x50 f32 (jit)               │  22.68μs │   2.04kw │   0.02x │      4017% │
│ Add 50x50 f32 (jit)                  │  23.46μs │   2.04kw │   0.02x │      4156% │
│ Square 100x100 f32 (jit)             │  23.59μs │   2.04kw │   0.02x │      4179% │
│ Mul 50x50 f32 (jit)                  │  24.65μs │   2.04kw │   0.02x │      4367% │
│ Square 50x50 f32 (forward)           │  32.31μs │   1.84kw │   0.02x │      5724% │
│ Sin 100x100 f32 (forward)            │  40.45μs │  945.00w │   0.01x │      7166% │
│ Max 100x100 f32 (forward)            │  46.54μs │  940.00w │   0.01x │      8244% │
│ Sum 100x100 f32 (forward)            │  51.88μs │  940.00w │   0.01x │      9189% │
│ Mean 100x100 f32 (forward)           │  51.93μs │   1.44kw │   0.01x │      9199% │
│ Sum 100x100 f32 (jit)                │  53.96μs │   1.14kw │   0.01x │      9559% │
│ Mean 100x100 f32 (jit)               │  55.95μs │   1.63kw │   0.01x │      9911% │
│ Exp 50x50 f32 (jit)                  │  80.00μs │   4.48kw │   0.01x │     14172% │
│ Matmul 100x100 f32 (forward)         │  83.40μs │   1.72kw │   0.01x │     14773% │
│ Mul 100x100 f32 (jit)                │  85.58μs │   2.04kw │   0.01x │     15160% │
│ Exp 50x50 f32 (forward)              │  93.64μs │   4.29kw │   0.01x │     16587% │
│ Cos 50x50 f32 (forward)              │  93.94μs │   4.88kw │   0.01x │     16642% │
│ Sqrt 50x50 f32 (forward)             │  99.05μs │   6.77kw │   0.01x │     17546% │
│ Matmul 100x100 f32 (jit)             │ 101.65μs │   1.91kw │   0.01x │     18006% │
│ BatchedMatmul 100x100 f32 (jit)      │ 112.54μs │   2.12kw │   0.01x │     19935% │
│ ReLU 50x50 f32 (jit)                 │ 121.40μs │   5.71kw │   0.00x │     21504% │
│ BatchedMatmul 100x100 f32 (forward)  │ 123.24μs │   1.92kw │   0.00x │     21832% │
│ Add 200x200 f32 (forward)            │ 128.17μs │   1.84kw │   0.00x │     22705% │
│ Mul 200x200 f32 (forward)            │ 137.68μs │   1.84kw │   0.00x │     24389% │
│ ReLU 50x50 f32 (forward)             │ 149.95μs │   5.51kw │   0.00x │     26562% │
│ Square 200x200 f32 (jit)             │ 161.96μs │   2.03kw │   0.00x │     28691% │
│ Add 200x200 f32 (jit)                │ 166.34μs │   2.04kw │   0.00x │     29467% │
│ Sigmoid 50x50 f32 (forward)          │ 173.67μs │   9.75kw │   0.00x │     30764% │
│ Max 200x200 f32 (forward)            │ 179.86μs │  940.00w │   0.00x │     31862% │
│ Square 200x200 f32 (forward)         │ 180.97μs │   1.84kw │   0.00x │     32058% │
│ Mul 200x200 f32 (jit)                │ 183.76μs │   2.04kw │   0.00x │     32551% │
│ Mean 200x200 f32 (forward)           │ 190.84μs │   1.44kw │   0.00x │     33806% │
│ Mean 200x200 f32 (jit)               │ 192.35μs │   1.63kw │   0.00x │     34073% │
│ Sum 200x200 f32 (jit)                │ 212.29μs │   1.14kw │   0.00x │     37606% │
│ Sum 200x200 f32 (forward)            │ 221.52μs │  940.00w │   0.00x │     39240% │
│ Exp 100x100 f32 (jit)                │ 230.44μs │   4.48kw │   0.00x │     40820% │
│ Sigmoid 50x50 f32 (jit)              │ 233.49μs │   9.94kw │   0.00x │     41362% │
│ Softmax 50x50 f32 (forward)          │ 244.76μs │  11.13kw │   0.00x │     43358% │
│ Exp 100x100 f32 (forward)            │ 247.39μs │   4.29kw │   0.00x │     43823% │
│ Broadcast 50x50 f32 (forward)        │ 257.33μs │  18.09kw │   0.00x │     45584% │
│ Softmax 50x50 f32 (jit)              │ 289.68μs │  11.33kw │   0.00x │     51316% │
│ Sin 200x200 f32 (forward)            │ 297.98μs │  938.00w │   0.00x │     52785% │
│ Sum 50x50 f32 (backward)             │ 332.15μs │  16.29kw │   0.00x │     58838% │
│ Log 50x50 f32 (forward)              │ 362.85μs │  13.53kw │   0.00x │     64276% │
│ ReLU 100x100 f32 (jit)               │ 369.04μs │   5.71kw │   0.00x │     65374% │
│ Cos 100x100 f32 (forward)            │ 411.48μs │   4.88kw │   0.00x │     72891% │
│ ReLU 100x100 f32 (forward)           │ 441.62μs │   5.51kw │   0.00x │     78230% │
│ Broadcast 100x100 f32 (forward)      │ 442.63μs │  19.32kw │   0.00x │     78409% │
│ Matmul 200x200 f32 (forward)         │ 453.12μs │   1.71kw │   0.00x │     80268% │
│ Tanh 50x50 f32 (jit)                 │ 532.29μs │  21.61kw │   0.00x │     94293% │
│ Square 50x50 f32 (backward)          │ 538.03μs │  26.09kw │   0.00x │     95308% │
│ Sin 50x50 f32 (backward)             │ 562.54μs │  27.70kw │   0.00x │     99650% │
│ Tanh 50x50 f32 (forward)             │ 593.31μs │  21.41kw │   0.00x │    105102% │
│ Sigmoid 100x100 f32 (forward)        │ 608.76μs │   9.75kw │   0.00x │    107838% │
│ Mean 50x50 f32 (backward)            │ 617.39μs │  22.23kw │   0.00x │    109367% │
│ Matmul 200x200 f32 (jit)             │ 623.39μs │   1.91kw │   0.00x │    110430% │
│ Matmul 50x50 f32 (backward)          │ 651.63μs │  30.54kw │   0.00x │    115432% │
│ Add 50x50 f32 (backward)             │ 668.82μs │  27.65kw │   0.00x │    118477% │
│ Transpose 50x50 f32 (backward)       │ 684.17μs │  21.86kw │   0.00x │    121197% │
│ Sigmoid 100x100 f32 (jit)            │ 709.56μs │   9.94kw │   0.00x │    125695% │
│ Mul 50x50 f32 (backward)             │ 821.24μs │  30.28kw │   0.00x │    145478% │
│ Sqrt 100x100 f32 (forward)           │ 835.57μs │   6.77kw │   0.00x │    148017% │
│ Reshape 50x50 f32 (backward)         │ 868.48μs │  19.48kw │   0.00x │    153846% │
│ Softmax 100x100 f32 (jit)            │ 967.53μs │  11.33kw │   0.00x │    171393% │
│ Mean 100x100 f32 (backward)          │ 983.66μs │  22.23kw │   0.00x │    174249% │
│ Broadcast 200x200 f32 (forward)      │ 993.60μs │  20.90kw │   0.00x │    176010% │
│ Softmax 100x100 f32 (forward)        │   1.01ms │  11.13kw │   0.00x │    178385% │
│ Tanh 100x100 f32 (jit)               │   1.01ms │  21.61kw │   0.00x │    178423% │
│ ReLU 200x200 f32 (forward)           │   1.05ms │   5.49kw │   0.00x │    185496% │
│ Tanh 100x100 f32 (forward)           │   1.12ms │  21.41kw │   0.00x │    198092% │
│ Sum 100x100 f32 (backward)           │   1.15ms │  16.29kw │   0.00x │    203768% │
│ Exp 50x50 f32 (backward)             │   1.17ms │  51.47kw │   0.00x │    206588% │
│ Log 100x100 f32 (forward)            │   1.22ms │  13.53kw │   0.00x │    215518% │
│ ReLU 200x200 f32 (jit)               │   1.23ms │   5.68kw │   0.00x │    217272% │
│ Broadcast 50x50 f32 (backward)       │   1.27ms │  49.57kw │   0.00x │    225102% │
│ Reshape 100x100 f32 (backward)       │   1.29ms │  19.48kw │   0.00x │    228412% │
│ Cos 50x50 f32 (backward)             │   1.42ms │  57.91kw │   0.00x │    250764% │
│ Exp 200x200 f32 (jit)                │   1.42ms │   4.46kw │   0.00x │    251354% │
│ Exp 200x200 f32 (forward)            │   1.61ms │   4.26kw │   0.00x │    285587% │
│ Square 100x100 f32 (backward)        │   1.80ms │  26.09kw │   0.00x │    319342% │
│ Mul 100x100 f32 (backward)           │   1.81ms │  30.28kw │   0.00x │    320242% │
│ ReLU 50x50 f32 (backward)            │   1.85ms │  62.40kw │   0.00x │    327076% │
│ Cos 200x200 f32 (forward)            │   2.09ms │   4.85kw │   0.00x │    370866% │
│ Transpose 100x100 f32 (backward)     │   2.12ms │  21.86kw │   0.00x │    375673% │
│ BatchedMatmul 50x50 f32 (backward)   │   2.17ms │  34.25kw │   0.00x │    383729% │
│ Sin 100x100 f32 (backward)           │   2.21ms │  27.70kw │   0.00x │    391908% │
│ BatchedMatmul 200x200 f32 (forward)  │   2.27ms │   1.92kw │   0.00x │    402744% │
│ Matmul 100x100 f32 (backward)        │   2.28ms │  30.54kw │   0.00x │    403305% │
│ Sqrt 50x50 f32 (backward)            │   2.42ms │  73.91kw │   0.00x │    428733% │
│ Add 100x100 f32 (backward)           │   2.45ms │  27.65kw │   0.00x │    433949% │
│ Log 50x50 f32 (backward)             │   2.58ms │ 119.50kw │   0.00x │    457452% │
│ Sigmoid 200x200 f32 (forward)        │   2.60ms │   9.70kw │   0.00x │    459957% │
│ Sqrt 200x200 f32 (forward)           │   2.74ms │   6.74kw │   0.00x │    485222% │
│ Sigmoid 50x50 f32 (backward)         │   2.76ms │  94.64kw │   0.00x │    489019% │
│ BatchedMatmul 200x200 f32 (jit)      │   2.86ms │   2.12kw │   0.00x │    507091% │
│ Broadcast 100x100 f32 (backward)     │   2.87ms │  50.79kw │   0.00x │    507989% │
│ Softmax 50x50 f32 (backward)         │   3.04ms │ 122.12kw │   0.00x │    539171% │
│ ReLU 100x100 f32 (backward)          │   3.35ms │  62.40kw │   0.00x │    594074% │
│ Sigmoid 200x200 f32 (jit)            │   3.71ms │   9.89kw │   0.00x │    657747% │
│ Exp 100x100 f32 (backward)           │   3.72ms │  51.47kw │   0.00x │    659399% │
│ Softmax 200x200 f32 (forward)        │   4.03ms │  11.08kw │   0.00x │    714073% │
│ Sqrt 100x100 f32 (backward)          │   4.26ms │  73.91kw │   0.00x │    753947% │
│ Cos 100x100 f32 (backward)           │   4.38ms │  57.91kw │   0.00x │    776491% │
│ Reshape 200x200 f32 (backward)       │   4.42ms │  19.42kw │   0.00x │    783713% │
│ Mean 200x200 f32 (backward)          │   4.61ms │  22.19kw │   0.00x │    817216% │
│ Sum 200x200 f32 (backward)           │   4.76ms │  16.24kw │   0.00x │    843141% │
│ Tanh 50x50 f32 (backward)            │   4.96ms │ 187.30kw │   0.00x │    878475% │
│ Log 200x200 f32 (forward)            │   5.64ms │  13.47kw │   0.00x │    998290% │
│ Sigmoid 100x100 f32 (backward)       │   5.73ms │  94.64kw │   0.00x │   1014569% │
│ Transpose 200x200 f32 (backward)     │   5.84ms │  21.80kw │   0.00x │   1034377% │
│ Tanh 200x200 f32 (forward)           │   6.30ms │  21.31kw │   0.00x │   1116231% │
│ Softmax 200x200 f32 (jit)            │   6.59ms │  11.27kw │   0.00x │   1167223% │
│ Softmax 100x100 f32 (backward)       │   7.49ms │ 122.12kw │   0.00x │   1326783% │
│ Log 100x100 f32 (backward)           │   7.89ms │ 119.50kw │   0.00x │   1396790% │
│ Mul 200x200 f32 (backward)           │   9.38ms │  30.18kw │   0.00x │   1662011% │
│ Square 200x200 f32 (backward)        │   9.46ms │  26.00kw │   0.00x │   1676619% │
│ Add 200x200 f32 (backward)           │  10.17ms │  27.56kw │   0.00x │   1801178% │
│ BatchedMatmul 100x100 f32 (backward) │  11.10ms │  34.15kw │   0.00x │   1967125% │
│ Broadcast 200x200 f32 (backward)     │  11.42ms │  52.29kw │   0.00x │   2023263% │
│ Tanh 100x100 f32 (backward)          │  11.99ms │ 187.30kw │   0.00x │   2123130% │
│ Sin 200x200 f32 (backward)           │  12.01ms │  27.59kw │   0.00x │   2128230% │
│ Tanh 200x200 f32 (jit)               │  15.37ms │  21.51kw │   0.00x │   2722486% │
│ ReLU 200x200 f32 (backward)          │  18.37ms │  62.20kw │   0.00x │   3254645% │
│ Exp 200x200 f32 (backward)           │  18.45ms │  51.30kw │   0.00x │   3268250% │
│ Cos 200x200 f32 (backward)           │  22.04ms │  57.69kw │   0.00x │   3904660% │
│ Sqrt 200x200 f32 (backward)          │  26.91ms │  73.64kw │   0.00x │   4766549% │
│ Sigmoid 200x200 f32 (backward)       │  28.39ms │  94.29kw │   0.00x │   5028247% │
│ Softmax 200x200 f32 (backward)       │  39.26ms │ 121.67kw │   0.00x │   6953879% │
│ Log 200x200 f32 (backward)           │  45.38ms │ 119.08kw │   0.00x │   8039613% │
│ Tanh 200x200 f32 (backward)          │  55.59ms │ 186.57kw │   0.00x │   9846939% │
│ Matmul 200x200 f32 (backward)        │ 198.05ms │  30.43kw │   0.00x │  35084126% │
│ BatchedMatmul 200x200 f32 (backward) │ 203.83ms │  34.15kw │   0.00x │  36106549% │
└──────────────────────────────────────┴──────────┴──────────┴─────────┴────────────┘

## System Information

- OCaml version: 5.3.0
- Rune version: Development
- Test sizes: 50, 100, 200
- Data types: Float32
- Measurements per benchmark: 10
- Warmup iterations: 3
