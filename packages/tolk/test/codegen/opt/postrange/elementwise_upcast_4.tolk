# tolk D280 golden=57bcfe5d2e5af5b10bd4ad8292105c27
0 Ops.CONST dtypes.weakint [] 256
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 1)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.AFTER dtypes.f32 [1, 7]
9 Ops.INDEX dtypes.f32 [8, 7]
10 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.INDEX dtypes.f32 [10, 7]
12 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
13 Ops.INDEX dtypes.f32 [12, 7]
14 Ops.ADD dtypes.f32 [11, 13]
15 Ops.SQRT dtypes.f32 [14]
16 Ops.CONST dtypes.weakfloat [] 1.0
17 Ops.ADD dtypes.f32 [11, 16]
18 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
19 Ops.MUL dtypes.f32 [17, 18]
20 Ops.EXP2 dtypes.f32 [19]
21 Ops.MUL dtypes.f32 [15, 20]
22 Ops.STORE dtypes.void [9, 21]
23 Ops.END dtypes.void [22, 3, 6]
24 Ops.SINK dtypes.void [23] KernelInfo(name="E_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)

