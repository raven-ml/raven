# tolk D141 golden=5f597010315cb3da59a3598fcdf97fad
0 Ops.CONST dtypes.weakint [] 9437184
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2359296
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 1536
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 3)
12 Ops.MUL dtypes.weakint [11, 10]
13 Ops.ADD dtypes.weakint [9, 12]
14 Ops.CONST dtypes.weakint [] 192
15 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 4)
16 Ops.CONST dtypes.weakint [] 8
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.RANGE dtypes.weakint [16] (AxisType.UPCAST, 6)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.ADD dtypes.weakint [13, 19]
21 Ops.INDEX dtypes.f32 [1, 20]
22 Ops.PARAM dtypes.bool [0] ParamArg(slot=1, dtype=dtypes.bool)
23 Ops.INDEX dtypes.bool [22, 20]
24 Ops.CONST dtypes.weakint [] 98304
25 Ops.PARAM dtypes.f32 [24] ParamArg(slot=2, dtype=dtypes.f32)
26 Ops.CONST dtypes.weakint [] 24576
27 Ops.MUL dtypes.weakint [7, 26]
28 Ops.CONST dtypes.weakint [] 16
29 Ops.MUL dtypes.weakint [11, 28]
30 Ops.ADD dtypes.weakint [27, 29]
31 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
32 Ops.MUL dtypes.weakint [31, 28]
33 Ops.RANGE dtypes.weakint [28] (AxisType.UPCAST, 7)
34 Ops.ADD dtypes.weakint [32, 33]
35 Ops.ADD dtypes.weakint [30, 34]
36 Ops.INDEX dtypes.f32 [25, 35]
37 Ops.PARAM dtypes.f32 [24] ParamArg(slot=3, dtype=dtypes.f32)
38 Ops.MUL dtypes.weakint [34, 10]
39 Ops.ADD dtypes.weakint [38, 27]
40 Ops.ADD dtypes.weakint [39, 19]
41 Ops.INDEX dtypes.f32 [37, 40]
42 Ops.MUL dtypes.f32 [36, 41]
43 Ops.REDUCE dtypes.f32 [42, 31, 33] (Ops.ADD, 0)
44 Ops.CONST dtypes.weakfloat [] -inf
45 Ops.WHERE dtypes.f32 [23, 43, 44]
46 Ops.STORE dtypes.void [21, 45]
47 Ops.END dtypes.void [46, 3, 6, 11, 15, 18]
48 Ops.SINK dtypes.void [47] KernelInfo(name="r_1536_192_4_8_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(0, AxisType.UPCAST))), optimized=True)

