# tolk D479 golden=0bf389ec4819fd10c8a3c74ca43c0ae6
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.GLOBAL, 1)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 4194304
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.CONST dtypes.weakint [] 1024
7 Ops.MUL dtypes.weakint [2, 6]
8 Ops.CONST dtypes.weakint [] 64
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
10 Ops.CONST dtypes.weakint [] 16
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.LOCAL, 2)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [7, 13]
15 Ops.INDEX dtypes.f32 [5, 14]
16 Ops.REDUCE dtypes.f32 [15, 9, 12] (Ops.ADD, 0)
17 Ops.STORE dtypes.void [3, 16]
18 Ops.END dtypes.void [17, 2]
19 Ops.SINK dtypes.void [18] KernelInfo(name="r_4096_16_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)),), optimized=True)

