# tolk D280 golden=d057f9c18eeaed904b4f3e265725cc75
0 Ops.CONST dtypes.weakint [] 24
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.INDEX dtypes.f32 [5, 2]
7 Ops.CONST dtypes.weakfloat [] 1.0
8 Ops.ADD dtypes.f32 [6, 7]
9 Ops.STORE dtypes.void [4, 8]
10 Ops.END dtypes.void [9, 2]
11 Ops.SINK dtypes.void [10] KernelInfo()

