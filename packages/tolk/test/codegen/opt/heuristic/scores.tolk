# tolk D280 golden=f08b9f8223d02eb636d4c8db00de73f8
0 Ops.CONST dtypes.weakint [] 131072
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 128
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
5 Ops.CONST dtypes.weakint [] 8
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 1)
7 Ops.AFTER dtypes.f32 [1, 3, 4, 6]
8 Ops.MUL dtypes.weakint [3, 2]
9 Ops.ADD dtypes.weakint [8, 4]
10 Ops.CONST dtypes.weakint [] 16384
11 Ops.MUL dtypes.weakint [6, 10]
12 Ops.ADD dtypes.weakint [9, 11]
13 Ops.INDEX dtypes.f32 [7, 12]
14 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 16
16 Ops.MUL dtypes.weakint [3, 15]
17 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.CONST dtypes.weakint [] 2048
20 Ops.MUL dtypes.weakint [6, 19]
21 Ops.ADD dtypes.weakint [18, 20]
22 Ops.INDEX dtypes.f32 [14, 21]
23 Ops.PARAM dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32)
24 Ops.MUL dtypes.weakint [4, 15]
25 Ops.ADD dtypes.weakint [24, 17]
26 Ops.ADD dtypes.weakint [25, 20]
27 Ops.INDEX dtypes.f32 [23, 26]
28 Ops.MUL dtypes.f32 [22, 27]
29 Ops.REDUCE dtypes.f32 [28, 17] (Ops.ADD, 0)
30 Ops.STORE dtypes.void [13, 29]
31 Ops.END dtypes.void [30, 6, 3, 4]
32 Ops.SINK dtypes.void [31] KernelInfo()

