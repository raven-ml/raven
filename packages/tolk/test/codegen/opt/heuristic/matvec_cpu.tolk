# tolk D280 golden=08046636f405a5870f86a939308cc507
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 256
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.AFTER dtypes.f32 [1, 7]
9 Ops.INDEX dtypes.f32 [8, 7]
10 Ops.CONST dtypes.weakint [] 4194304
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 4096
13 Ops.MUL dtypes.weakint [7, 12]
14 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [13, 17]
19 Ops.INDEX dtypes.f32 [11, 18]
20 Ops.PARAM dtypes.f32 [12] ParamArg(slot=2, dtype=dtypes.f32)
21 Ops.INDEX dtypes.f32 [20, 17]
22 Ops.MUL dtypes.f32 [19, 21]
23 Ops.REDUCE dtypes.f32 [22, 14, 16] (Ops.ADD, 0)
24 Ops.STORE dtypes.void [9, 23]
25 Ops.END dtypes.void [24, 3, 6]
26 Ops.SINK dtypes.void [25] KernelInfo(name="r_256_4_1024_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True)

