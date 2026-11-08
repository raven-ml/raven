# tolk D141 D280 golden=d4f28d986a84e38e4e5c368953741ec0
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
6 Ops.MUL dtypes.weakint [5, 4]
7 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
8 Ops.ADD dtypes.weakint [6, 7]
9 Ops.CONST dtypes.weakint [] 16
10 Ops.MUL dtypes.weakint [8, 9]
11 Ops.RANGE dtypes.weakint [9] (AxisType.UPCAST, 3)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.AFTER dtypes.f32 [1, 3, 12]
14 Ops.MUL dtypes.weakint [3, 2]
15 Ops.ADD dtypes.weakint [14, 12]
16 Ops.INDEX dtypes.f32 [13, 15]
17 Ops.CONST dtypes.weakint [] 65536
18 Ops.PARAM dtypes.bf16 [17] ParamArg(slot=1, dtype=dtypes.bf16)
19 Ops.CONST dtypes.weakint [] 1024
20 Ops.MUL dtypes.weakint [3, 19]
21 Ops.RANGE dtypes.weakint [19] (AxisType.WEAK, 0)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.INDEX dtypes.bf16 [18, 22]
24 Ops.CAST dtypes.f32 [23] dtypes.f32
25 Ops.PARAM dtypes.bf16 [17] ParamArg(slot=2, dtype=dtypes.bf16)
26 Ops.MUL dtypes.weakint [21, 2]
27 Ops.ADD dtypes.weakint [26, 12]
28 Ops.INDEX dtypes.bf16 [25, 27]
29 Ops.CAST dtypes.f32 [28] dtypes.f32
30 Ops.MUL dtypes.f32 [24, 29]
31 Ops.REDUCE dtypes.f32 [30, 21] (Ops.ADD, 0)
32 Ops.STORE dtypes.void [16, 31]
33 Ops.END dtypes.void [32, 3, 5, 7, 11]
34 Ops.SINK dtypes.void [33] KernelInfo(name="r_64_2_16_2_1024", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST))), optimized=True)

