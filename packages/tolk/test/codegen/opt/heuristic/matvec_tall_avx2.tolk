# tolk D141 D188 golden=ff0f02ba70cf0fd5df940ec30baea64f
0 Ops.CONST dtypes.weakint [] 8
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 0]
5 Ops.RANGE dtypes.weakint [0] (AxisType.UPCAST, 2)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.INDEX dtypes.f32 [1, 6]
8 Ops.CONST dtypes.weakint [] 8388608
9 Ops.PARAM dtypes.f32 [8] ParamArg(slot=1, dtype=dtypes.f32)
10 Ops.CONST dtypes.weakint [] 1048576
11 Ops.MUL dtypes.weakint [6, 10]
12 Ops.CONST dtypes.weakint [] 262144
13 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
14 Ops.CONST dtypes.weakint [] 4
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 3)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [11, 17]
19 Ops.INDEX dtypes.f32 [9, 18]
20 Ops.PARAM dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32)
21 Ops.INDEX dtypes.f32 [20, 17]
22 Ops.MUL dtypes.f32 [19, 21]
23 Ops.REDUCE dtypes.f32 [22, 13, 16] (Ops.ADD, 0)
24 Ops.STORE dtypes.void [7, 23]
25 Ops.END dtypes.void [24, 3, 5]
26 Ops.SINK dtypes.void [25] KernelInfo(name="r_8_262144_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST))), optimized=True)

