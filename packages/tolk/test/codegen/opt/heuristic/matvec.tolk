# tolk D280 golden=8f93c8c4609e80d2a0512af9dd5b310a
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.CONST dtypes.weakint [] 4194304
6 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32)
7 Ops.CONST dtypes.weakint [] 4096
8 Ops.MUL dtypes.weakint [2, 7]
9 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 0)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.INDEX dtypes.f32 [6, 10]
12 Ops.PARAM dtypes.f32 [7] ParamArg(slot=2, dtype=dtypes.f32)
13 Ops.INDEX dtypes.f32 [12, 9]
14 Ops.MUL dtypes.f32 [11, 13]
15 Ops.REDUCE dtypes.f32 [14, 9] (Ops.ADD, 0)
16 Ops.STORE dtypes.void [4, 15]
17 Ops.END dtypes.void [16, 2]
18 Ops.SINK dtypes.void [17] KernelInfo()

