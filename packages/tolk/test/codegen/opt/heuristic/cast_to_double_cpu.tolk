# tolk D141 golden=909ff5636dc06033da509476145bb069
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f64 [0] ParamArg(slot=0, dtype=dtypes.f64)
2 Ops.CONST dtypes.weakint [] 512
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 1)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.f64 [1, 7]
9 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
10 Ops.INDEX dtypes.f32 [9, 7]
11 Ops.CAST dtypes.f64 [10] dtypes.f64
12 Ops.CONST dtypes.weakfloat [] 1.0
13 Ops.ADD dtypes.f64 [11, 12]
14 Ops.STORE dtypes.void [8, 13]
15 Ops.END dtypes.void [14, 3, 6]
16 Ops.SINK dtypes.void [15] KernelInfo(name="E_512_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)),), optimized=True)

