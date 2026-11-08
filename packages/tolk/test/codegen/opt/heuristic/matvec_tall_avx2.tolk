# tolk D141 D280 golden=ff0f02ba70cf0fd5df940ec30baea64f
0 Ops.CONST dtypes.weakint [] 8
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.AFTER dtypes.f32 [1, 7]
9 Ops.INDEX dtypes.f32 [8, 7]
10 Ops.CONST dtypes.weakint [] 8388608
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 1048576
13 Ops.MUL dtypes.weakint [7, 12]
14 Ops.CONST dtypes.weakint [] 262144
15 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 0)
16 Ops.MUL dtypes.weakint [15, 4]
17 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.ADD dtypes.weakint [13, 18]
20 Ops.INDEX dtypes.f32 [11, 19]
21 Ops.PARAM dtypes.f32 [12] ParamArg(slot=2, dtype=dtypes.f32)
22 Ops.INDEX dtypes.f32 [21, 18]
23 Ops.MUL dtypes.f32 [20, 22]
24 Ops.REDUCE dtypes.f32 [23, 15, 17] (Ops.ADD, 0)
25 Ops.STORE dtypes.void [9, 24]
26 Ops.END dtypes.void [25, 3, 6]
27 Ops.SINK dtypes.void [26] KernelInfo(name="r_2_4_262144_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True)

