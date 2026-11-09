# tolk D141 golden=be2d7f67895ff1bac0ce64d7021bb54a
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 16
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 100544
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 1571
12 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
13 Ops.MUL dtypes.weakint [12, 0]
14 Ops.ADD dtypes.weakint [13, 7]
15 Ops.INDEX dtypes.f32 [10, 14]
16 Ops.REDUCE dtypes.f32 [15, 12] (Ops.ADD, 0)
17 Ops.STORE dtypes.void [8, 16]
18 Ops.END dtypes.void [17, 3, 6]
19 Ops.SINK dtypes.void [18] KernelInfo(name="r_4_16_1571", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)),), optimized=True)

