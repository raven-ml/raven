# tolk D188 golden=08046636f405a5870f86a939308cc507
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 128
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 4194304
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 4096
16 Ops.MUL dtypes.weakint [11, 15]
17 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
18 Ops.MUL dtypes.weakint [17, 8]
19 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 4)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.ADD dtypes.weakint [16, 20]
22 Ops.INDEX dtypes.f32 [14, 21]
23 Ops.PARAM dtypes.f32 [15] ParamArg(slot=2, dtype=dtypes.f32)
24 Ops.INDEX dtypes.f32 [23, 20]
25 Ops.MUL dtypes.f32 [22, 24]
26 Ops.REDUCE dtypes.f32 [25, 17, 19] (Ops.ADD, 0)
27 Ops.STORE dtypes.void [12, 26]
28 Ops.END dtypes.void [27, 3, 6, 10]
29 Ops.SINK dtypes.void [28] KernelInfo(name="r_128_4_2_1024_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(4, AxisType.UPCAST))), optimized=True)

