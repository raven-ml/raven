# Nx Benchmarks

┌───────────────────────────────────┬─────────────────┬──────────────┬────────────┐
│ Name                              │        Time/Run │      mWd/Run │ vs Fastest │
├───────────────────────────────────┼─────────────────┼──────────────┼────────────┤
│ Sqrt on 50x50 float32             │     49_807.04ns │      160.77w │    100.00% │
│ Sqrt on 50x50 float64             │     50_403.56ns │      160.77w │    101.20% │
│ Sum on 50x50 float64              │     50_851.41ns │     7098.60w │    102.10% │
│ Multiplication on 50x50 float64   │     51_725.61ns │      166.53w │    103.85% │
│ Square on 50x50 float32           │     52_383.83ns │      166.53w │    105.17% │
│ Addition on 50x50 float32         │     52_482.01ns │      166.53w │    105.37% │
│ Multiplication on 50x50 float32   │     54_258.46ns │      166.53w │    108.94% │
│ Addition on 50x50 float64         │     54_882.09ns │      166.53w │    110.19% │
│ Square on 50x50 float64           │     56_399.55ns │      166.53w │    113.24% │
│ Sum on 50x50 float32              │     60_956.62ns │     7098.60w │    122.39% │
│ Sqrt on 100x100 float32           │     61_642.89ns │      162.31w │    123.76% │
│ Addition on 100x100 float32       │     62_670.43ns │      168.13w │    125.83% │
│ Multiplication on 100x100 float32 │     64_776.93ns │      168.13w │    130.06% │
│ Square on 100x100 float32         │     67_215.34ns │      168.13w │    134.95% │
│ Sum on 100x100 float32            │     72_290.38ns │    36769.29w │    145.14% │
│ Sum on 100x100 float64            │     74_898.03ns │    36735.27w │    150.38% │
│ Sqrt on 100x100 float64           │     95_314.87ns │       62.02w │    191.37% │
│ Addition on 100x100 float64       │     96_739.75ns │        0.00w │    194.23% │
│ Multiplication on 100x100 float64 │    127_779.22ns │      188.89w │    256.55% │
│ Exp on 50x50 float64              │    135_551.29ns │    57886.56w │    272.15% │
│ Square on 100x100 float64         │    139_660.72ns │      192.54w │    280.40% │
│ Exp on 50x50 float32              │    145_675.63ns │    51446.78w │    292.48% │
│ Sqrt on 500x500 float32           │    197_951.76ns │        0.00w │    397.44% │
│ Exp on 100x100 float32            │    232_541.37ns │   273010.79w │    466.88% │
│ Multiplication on 500x500 float32 │    268_981.57ns │        0.00w │    540.05% │
│ Addition on 500x500 float32       │    271_827.46ns │        0.00w │    545.76% │
│ Square on 500x500 float32         │    282_616.84ns │      120.28w │    567.42% │
│ Addition on 500x500 float64       │    320_271.84ns │      120.42w │    643.03% │
│ Sqrt on 500x500 float64           │    323_257.14ns │      189.11w │    649.02% │
│ Square on 500x500 float64         │    378_285.68ns │      177.19w │    759.50% │
│ Exp on 100x100 float64            │    387_614.77ns │   281174.08w │    778.23% │
│ Multiplication on 500x500 float64 │    397_289.95ns │      175.38w │    797.66% │
│ Sum on 500x500 float32            │    491_972.89ns │   976917.69w │    987.76% │
│ Sum on 500x500 float64            │    492_965.02ns │   976637.06w │    989.75% │
│ MatMul on 50x50 float32           │  3_436_724.34ns │  5866021.22w │   6900.08% │
│ MatMul on 50x50 float64           │  3_921_270.37ns │  5683871.89w │   7872.92% │
│ Exp on 500x500 float64            │  4_340_012.87ns │  6965829.22w │   8713.65% │
│ Exp on 500x500 float32            │ 11_564_016.34ns │  7361698.22w │            │
│ MatMul on 100x100 float64         │ 46_678_066.25ns │ 49008947.67w │            │
│ MatMul on 100x100 float32         │ 52_562_952.04ns │ 48904441.33w │            │
└───────────────────────────────────┴─────────────────┴──────────────┴────────────┘
