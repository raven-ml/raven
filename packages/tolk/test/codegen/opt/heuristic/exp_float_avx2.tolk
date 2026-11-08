# tolk D280 golden=a7c28de68db68a153c59479c387b66f8
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 256
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 16
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 1)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.AFTER dtypes.f32 [1, 7]
9 Ops.INDEX dtypes.f32 [8, 7]
10 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.INDEX dtypes.f32 [10, 7]
12 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
13 Ops.MUL dtypes.f32 [11, 12]
14 Ops.EXP2 dtypes.f32 [13]
15 Ops.STORE dtypes.void [9, 14]
16 Ops.END dtypes.void [15, 3, 6]
17 Ops.SINK dtypes.void [16] KernelInfo(name="E_256_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)),), optimized=True)

