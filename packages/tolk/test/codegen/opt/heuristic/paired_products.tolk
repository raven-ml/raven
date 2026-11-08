# tolk D280 golden=87b1449f113a7258729d42d827e1423a
0 Ops.CONST dtypes.weakint [] 32768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
5 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
6 Ops.AFTER dtypes.f32 [1, 3, 4, 5]
7 Ops.MUL dtypes.weakint [3, 2]
8 Ops.ADD dtypes.weakint [7, 4]
9 Ops.CONST dtypes.weakint [] 1024
10 Ops.MUL dtypes.weakint [5, 9]
11 Ops.ADD dtypes.weakint [8, 10]
12 Ops.INDEX dtypes.f32 [6, 11]
13 Ops.CONST dtypes.weakint [] 16384
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 16
16 Ops.MUL dtypes.weakint [3, 15]
17 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.CONST dtypes.weakint [] 512
20 Ops.MUL dtypes.weakint [5, 19]
21 Ops.ADD dtypes.weakint [18, 20]
22 Ops.INDEX dtypes.f32 [14, 21]
23 Ops.PARAM dtypes.f32 [13] ParamArg(slot=2, dtype=dtypes.f32)
24 Ops.MUL dtypes.weakint [4, 15]
25 Ops.ADD dtypes.weakint [24, 17]
26 Ops.MUL dtypes.weakint [3, 19]
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.INDEX dtypes.f32 [23, 27]
29 Ops.MUL dtypes.f32 [22, 28]
30 Ops.PARAM dtypes.f32 [13] ParamArg(slot=3, dtype=dtypes.f32)
31 Ops.ADD dtypes.weakint [25, 20]
32 Ops.INDEX dtypes.f32 [30, 31]
33 Ops.PARAM dtypes.f32 [13] ParamArg(slot=4, dtype=dtypes.f32)
34 Ops.INDEX dtypes.f32 [33, 21]
35 Ops.MUL dtypes.f32 [32, 34]
36 Ops.ADD dtypes.f32 [29, 35]
37 Ops.REDUCE dtypes.f32 [36, 17] (Ops.ADD, 0)
38 Ops.STORE dtypes.void [12, 37]
39 Ops.END dtypes.void [38, 5, 3, 4]
40 Ops.SINK dtypes.void [39] KernelInfo()

