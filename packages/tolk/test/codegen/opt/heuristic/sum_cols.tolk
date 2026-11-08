# tolk D280 golden=7a877b5ada10de0c9e89d92e9f45f30d
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.CONST dtypes.weakint [] 262144
6 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32)
7 Ops.CONST dtypes.weakint [] 64
8 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 0)
9 Ops.MUL dtypes.weakint [8, 0]
10 Ops.ADD dtypes.weakint [9, 2]
11 Ops.INDEX dtypes.f32 [6, 10]
12 Ops.REDUCE dtypes.f32 [11, 8] (Ops.ADD, 0)
13 Ops.STORE dtypes.void [4, 12]
14 Ops.END dtypes.void [13, 2]
15 Ops.SINK dtypes.void [14] KernelInfo()

