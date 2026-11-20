# tolk D141 D188 golden=ff0f02ba70cf0fd5df940ec30baea64f
0 Ops.CONST dtypes.weakint [] 8
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
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
13 Ops.CONST dtypes.weakint [] 8388608
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 1048576
16 Ops.MUL dtypes.weakint [11, 15]
17 Ops.CONST dtypes.weakint [] 262144
18 Ops.RANGE dtypes.weakint [17] (AxisType.WEAK, 0)
19 Ops.MUL dtypes.weakint [18, 8]
20 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 4)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.ADD dtypes.weakint [16, 21]
23 Ops.INDEX dtypes.f32 [14, 22]
24 Ops.PARAM dtypes.f32 [15] ParamArg(slot=2, dtype=dtypes.f32)
25 Ops.INDEX dtypes.f32 [24, 21]
26 Ops.MUL dtypes.f32 [23, 25]
27 Ops.REDUCE dtypes.f32 [26, 18, 20] (Ops.ADD, 0)
28 Ops.STORE dtypes.void [12, 27]
29 Ops.END dtypes.void [28, 3, 6, 10]
30 Ops.SINK dtypes.void [29] KernelInfo(name="r_4_2_262144_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True)

