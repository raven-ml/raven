# tolk D280 golden=307939c7875ae09ed7b40d79881671a9
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
5 Ops.AFTER dtypes.f32 [1, 3, 4]
6 Ops.MUL dtypes.weakint [3, 2]
7 Ops.ADD dtypes.weakint [6, 4]
8 Ops.INDEX dtypes.f32 [5, 7]
9 Ops.CONST dtypes.weakint [] 1048576
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 128
12 Ops.MUL dtypes.weakint [4, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 1)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
16 Ops.CONST dtypes.weakint [] 1024
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [14, 17]
19 Ops.CONST dtypes.weakint [] 131072
20 Ops.MUL dtypes.weakint [3, 19]
21 Ops.ADD dtypes.weakint [18, 20]
22 Ops.INDEX dtypes.f32 [10, 21]
23 Ops.REDUCE dtypes.f32 [22, 15, 13] (Ops.ADD, 0)
24 Ops.STORE dtypes.void [8, 23]
25 Ops.END dtypes.void [24, 3, 4]
26 Ops.SINK dtypes.void [25] KernelInfo()

