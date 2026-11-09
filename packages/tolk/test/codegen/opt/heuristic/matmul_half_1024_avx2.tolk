# tolk D141 golden=bffec475aec1c7390b797023cdfe5d3c
0 Ops.CONST dtypes.weakint [] 1048576
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 256
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 1024
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 64
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.CONST dtypes.weakint [] 16
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
19 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
20 Ops.ADD dtypes.weakint [9, 19]
21 Ops.INDEX dtypes.f16 [18, 20]
22 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
23 Ops.MUL dtypes.weakint [19, 8]
24 Ops.ADD dtypes.weakint [23, 15]
25 Ops.INDEX dtypes.f16 [22, 24]
26 Ops.MUL dtypes.f16 [21, 25]
27 Ops.CAST dtypes.f32 [26] dtypes.f32
28 Ops.REDUCE dtypes.f32 [27, 19] (Ops.ADD, 0)
29 Ops.STORE dtypes.void [17, 28]
30 Ops.END dtypes.void [29, 3, 6, 11, 14]
31 Ops.SINK dtypes.void [30] KernelInfo(name="r_256_64_16_4_1024", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))), optimized=True)

