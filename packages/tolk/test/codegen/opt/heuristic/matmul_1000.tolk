# tolk D280 golden=06f5a05dc41b2a6d8ff326237c6ab74c
0 Ops.CONST dtypes.weakint [] 1000000
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1000
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
5 Ops.AFTER dtypes.f32 [1, 3, 4]
6 Ops.MUL dtypes.weakint [3, 2]
7 Ops.ADD dtypes.weakint [6, 4]
8 Ops.INDEX dtypes.f32 [5, 7]
9 Ops.CONST dtypes.weakint [] 16000
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 16
12 Ops.MUL dtypes.weakint [3, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.INDEX dtypes.f32 [10, 14]
16 Ops.PARAM dtypes.f32 [9] ParamArg(slot=2, dtype=dtypes.f32)
17 Ops.MUL dtypes.weakint [13, 2]
18 Ops.ADD dtypes.weakint [17, 4]
19 Ops.INDEX dtypes.f32 [16, 18]
20 Ops.MUL dtypes.f32 [15, 19]
21 Ops.REDUCE dtypes.f32 [20, 13] (Ops.ADD, 0)
22 Ops.STORE dtypes.void [8, 21]
23 Ops.END dtypes.void [22, 3, 4]
24 Ops.SINK dtypes.void [23] KernelInfo()

