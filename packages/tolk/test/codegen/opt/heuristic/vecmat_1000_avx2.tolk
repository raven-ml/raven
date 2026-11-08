# tolk D141 D280 golden=527af719c5462caa92ea0a9de5c83977
0 Ops.CONST dtypes.weakint [] 1000
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 125
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.AFTER dtypes.f32 [1, 7]
9 Ops.INDEX dtypes.f32 [8, 7]
10 Ops.CONST dtypes.weakint [] 4096
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 1024
13 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
14 Ops.CONST dtypes.weakint [] 4
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 3)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.INDEX dtypes.f32 [11, 17]
19 Ops.CONST dtypes.weakint [] 4096000
20 Ops.PARAM dtypes.f32 [19] ParamArg(slot=2, dtype=dtypes.f32)
21 Ops.MUL dtypes.weakint [17, 0]
22 Ops.ADD dtypes.weakint [21, 7]
23 Ops.INDEX dtypes.f32 [20, 22]
24 Ops.MUL dtypes.f32 [18, 23]
25 Ops.REDUCE dtypes.f32 [24, 13, 16] (Ops.ADD, 0)
26 Ops.STORE dtypes.void [9, 25]
27 Ops.END dtypes.void [26, 3, 6]
28 Ops.SINK dtypes.void [27] KernelInfo(name="r_125_8_1024_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True)

