# tolk D479 D487 golden=0bf389ec4819fd10c8a3c74ca43c0ae6
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.GLOBAL, 1)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 4194304
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.CONST dtypes.weakint [] 1024
7 Ops.MUL dtypes.weakint [2, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
10 Ops.MUL dtypes.weakint [9, 8]
11 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 2)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.ADD dtypes.weakint [7, 12]
14 Ops.INDEX dtypes.f32 [5, 13]
15 Ops.REDUCE dtypes.f32 [14, 9, 11] (Ops.ADD, 0)
16 Ops.STORE dtypes.void [3, 15]
17 Ops.END dtypes.void [16, 2]
18 Ops.SINK dtypes.void [17] KernelInfo(name="r_4096_32_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(32, AxisType.LOCAL)),), optimized=True)

