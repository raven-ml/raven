# NumPy Benchmarks

┌─────────────────────────────────────┬──────────────┬────────────┐
│ Name                                │ Time/Run (ns) │ Percentage │
├─────────────────────────────────────┼──────────────┼────────────┤
│ Square on 100x100 float32           │   1_083.36ns │    100.00% │
│ Sqrt on 50x50 float32               │   1_124.96ns │    103.84% │
│ Square on 50x50 float32             │   1_208.31ns │    111.53% │
│ Sqrt on 50x50 float64               │   1_278.09ns │    117.97% │
│ Multiplication on 100x100 float32   │   1_305.64ns │    120.52% │
│ Multiplication on 50x50 float32     │   1_333.03ns │    123.05% │
│ Multiplication on 50x50 float64     │   1_375.41ns │    126.96% │
│ Square on 50x50 float64             │   1_486.31ns │    137.19% │
│ Square on 100x100 float64           │   1_708.12ns │    157.67% │
│ Addition on 50x50 float64           │   1_708.36ns │    157.69% │
│ Sqrt on 100x100 float32             │   2_041.61ns │    188.45% │
│ Addition on 50x50 float32           │   2_625.32ns │    242.33% │
│ Addition on 100x100 float32         │   2_638.75ns │    243.57% │
│ Multiplication on 100x100 float64   │   3_069.64ns │    283.34% │
│ Addition on 100x100 float64         │   3_291.68ns │    303.84% │
│ Sum on 50x50 float64                │   3_333.59ns │    307.71% │
│ Sqrt on 100x100 float64             │   3_652.96ns │    337.19% │
│ Sum on 100x100 float64              │   3_846.98ns │    355.10% │
│ Sum on 100x100 float32              │   4_333.29ns │    399.99% │
│ Exp on 50x50 float32                │   5_194.30ns │    479.46% │
│ Sum on 50x50 float32                │   6_416.66ns │    592.29% │
│ Exp on 50x50 float64                │   6_458.33ns │    596.14% │
│ MatMul on 50x50 float64             │   6_999.98ns │    646.14% │
│ MatMul on 100x100 float32           │   8_250.04ns │    761.52% │
│ MatMul on 100x100 float64           │  11_319.06ns │   1044.81% │
│ Exp on 100x100 float32              │  16_778.01ns │   1548.70% │
│ MatMul on 50x50 float32             │  19_527.97ns │   1802.54% │
│ Square on 500x500 float32           │  21_972.00ns │   2028.13% │
│ Exp on 100x100 float64              │  23_763.63ns │   2193.51% │
│ Multiplication on 500x500 float32   │  32_416.62ns │   2992.23% │
│ Sqrt on 500x500 float32             │  41_833.30ns │   3861.44% │
│ Sum on 500x500 float32              │  45_986.07ns │   4244.76% │
│ Addition on 500x500 float32         │  48_472.62ns │   4474.28% │
│ Sum on 500x500 float64              │  55_291.30ns │   5103.68% │
│ Square on 500x500 float64           │ 148_153.01ns │  13675.31% │
│ Addition on 500x500 float64         │ 160_666.65ns │  14830.39% │
│ Sqrt on 500x500 float64             │ 164_055.34ns │  15143.18% │
│ Multiplication on 500x500 float64   │ 164_403.04ns │  15175.28% │
│ Exp on 500x500 float32              │ 422_138.95ns │  38965.68% │
│ Exp on 500x500 float64              │ 699_055.69ns │  64526.57% │
└─────────────────────────────────────┴──────────────┴────────────┘
