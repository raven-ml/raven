# tolk D280 golden=f5aec437aae24b6c8e501d11b50e64df
0 Ops.CONST dtypes.weakint [] 7
1 Ops.PARAM dtypes.bool [0] ParamArg(slot=0, dtype=dtypes.bool)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.bool [1, 2]
4 Ops.INDEX dtypes.bool [3, 2]
5 Ops.CONST dtypes.weakfloat [] 0.0
6 Ops.CONST dtypes.weakint [] 35
7 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32)
8 Ops.CONST dtypes.weakint [] 5
9 Ops.MUL dtypes.weakint [2, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [7, 11]
13 Ops.CMPLT dtypes.bool [5, 12]
14 Ops.REDUCE dtypes.bool [13, 10] (Ops.MUL, 0)
15 Ops.STORE dtypes.void [4, 14]
16 Ops.END dtypes.void [15, 2]
17 Ops.SINK dtypes.void [16] KernelInfo()

