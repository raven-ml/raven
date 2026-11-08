# tolk D280 golden=03b43a8afcbb25e283d460bee40118d2
0 Ops.CONST dtypes.weakint [] 8
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CAST dtypes.i32 [0] dtypes.i32
3 Ops.RANGE dtypes.i32 [2] (AxisType.GLOBAL, 0)
4 Ops.AFTER dtypes.f32 [1, 3]
5 Ops.INDEX dtypes.f32 [4, 3]
6 Ops.CONST dtypes.weakfloat [] 1.0
7 Ops.CAST dtypes.f32 [6] dtypes.f32
8 Ops.STORE dtypes.void [5, 7]
9 Ops.END dtypes.void [8, 3]
10 Ops.SINK dtypes.void [9] KernelInfo()

