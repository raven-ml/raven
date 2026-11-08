# tolk D280 golden=14765dde4c891a04b0c7c334c19a5261
0 Ops.CONST dtypes.weakint [] 2048
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.GLOBAL, 1)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.CONST dtypes.weakint [] 32768
6 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32)
7 Ops.CONST dtypes.weakint [] 16
8 Ops.MUL dtypes.weakint [2, 7]
9 Ops.CONST dtypes.weakint [] 1
10 Ops.RANGE dtypes.weakint [9] (AxisType.WEAK, 0)
11 Ops.MUL dtypes.weakint [10, 7]
12 Ops.RANGE dtypes.weakint [7] (AxisType.LOCAL, 2)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [8, 13]
15 Ops.INDEX dtypes.f32 [6, 14]
16 Ops.REDUCE dtypes.f32 [15, 10, 12] (Ops.MAX, 0)
17 Ops.STORE dtypes.void [4, 16]
18 Ops.END dtypes.void [17, 2]
19 Ops.SINK dtypes.void [18] KernelInfo(name="r_2048_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)),), optimized=True)

