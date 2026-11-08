# tolk D29 D280 golden=e91d4d62359283ce87fe3246e1b5ca9f
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
5 Ops.AFTER dtypes.f32 [1, 3, 4]
6 Ops.MUL dtypes.weakint [3, 2]
7 Ops.ADD dtypes.weakint [6, 4]
8 Ops.INDEX dtypes.f32 [5, 7]
9 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
10 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
11 Ops.ADD dtypes.weakint [6, 10]
12 Ops.INDEX dtypes.f16 [9, 11]
13 Ops.CAST dtypes.f32 [12] dtypes.f32
14 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
15 Ops.MUL dtypes.weakint [10, 2]
16 Ops.ADD dtypes.weakint [15, 4]
17 Ops.INDEX dtypes.f16 [14, 16]
18 Ops.CAST dtypes.f32 [17] dtypes.f32
19 Ops.MUL dtypes.f32 [13, 18]
20 Ops.REDUCE dtypes.f32 [19, 10] (Ops.ADD, 0)
21 Ops.STORE dtypes.void [8, 20]
22 Ops.END dtypes.void [21, 3, 4]
23 Ops.SINK dtypes.void [22] KernelInfo()

