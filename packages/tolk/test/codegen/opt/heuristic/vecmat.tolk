# tolk D280 golden=8e53e1f55cd37a69f76178247685dd63
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.CONST dtypes.weakint [] 4096
6 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32)
7 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 0)
8 Ops.INDEX dtypes.f32 [6, 7]
9 Ops.CONST dtypes.weakint [] 4194304
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=2, dtype=dtypes.f32)
11 Ops.MUL dtypes.weakint [7, 0]
12 Ops.ADD dtypes.weakint [11, 2]
13 Ops.INDEX dtypes.f32 [10, 12]
14 Ops.MUL dtypes.f32 [8, 13]
15 Ops.REDUCE dtypes.f32 [14, 7] (Ops.ADD, 0)
16 Ops.STORE dtypes.void [4, 15]
17 Ops.END dtypes.void [16, 2]
18 Ops.SINK dtypes.void [17] KernelInfo()

