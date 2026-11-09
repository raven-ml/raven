# tolk D141 golden=ff0f02ba70cf0fd5df940ec30baea64f
0 Ops.CONST dtypes.weakint [] 8
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 8388608
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 1048576
12 Ops.MUL dtypes.weakint [7, 11]
13 Ops.CONST dtypes.weakint [] 262144
14 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0)
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [12, 17]
19 Ops.INDEX dtypes.f32 [10, 18]
20 Ops.PARAM dtypes.f32 [11] ParamArg(slot=2, dtype=dtypes.f32)
21 Ops.INDEX dtypes.f32 [20, 17]
22 Ops.MUL dtypes.f32 [19, 21]
23 Ops.REDUCE dtypes.f32 [22, 14, 16] (Ops.ADD, 0)
24 Ops.STORE dtypes.void [8, 23]
25 Ops.END dtypes.void [24, 3, 6]
26 Ops.SINK dtypes.void [25] KernelInfo(name="r_2_4_262144_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True)

