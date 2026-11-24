# tolk D486 D503 golden=051db0a3fd9f489e91b2aa54a56f465b
0 Ops.CONST dtypes.weakint [] 9437184
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2359296
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 64
11 Ops.RANGE dtypes.weakint [10] (AxisType.GLOBAL, 3)
12 Ops.CONST dtypes.weakint [] 8
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.LOCAL, 9)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.CONST dtypes.weakint [] 3
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.RANGE dtypes.weakint [16] (AxisType.UPCAST, 7)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.CONST dtypes.weakint [] 1536
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.ADD dtypes.weakint [9, 21]
23 Ops.CONST dtypes.weakint [] 32
24 Ops.RANGE dtypes.weakint [23] (AxisType.GLOBAL, 4)
25 Ops.CONST dtypes.weakint [] 16
26 Ops.MUL dtypes.weakint [24, 25]
27 Ops.RANGE dtypes.weakint [25] (AxisType.LOCAL, 8)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.MUL dtypes.weakint [28, 16]
30 Ops.RANGE dtypes.weakint [16] (AxisType.UPCAST, 6)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.ADD dtypes.weakint [22, 31]
33 Ops.INDEX dtypes.f32 [1, 32]
34 Ops.PARAM dtypes.bool [0] ParamArg(slot=1, dtype=dtypes.bool)
35 Ops.INDEX dtypes.bool [34, 32]
36 Ops.CONST dtypes.weakint [] 98304
37 Ops.PARAM dtypes.f32 [36] ParamArg(slot=2, dtype=dtypes.f32)
38 Ops.CONST dtypes.weakint [] 24576
39 Ops.MUL dtypes.weakint [7, 38]
40 Ops.MUL dtypes.weakint [19, 25]
41 Ops.ADD dtypes.weakint [39, 40]
42 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 0)
43 Ops.ADD dtypes.weakint [41, 42]
44 Ops.INDEX dtypes.f32 [37, 43]
45 Ops.PARAM dtypes.f32 [36] ParamArg(slot=3, dtype=dtypes.f32)
46 Ops.MUL dtypes.weakint [42, 20]
47 Ops.ADD dtypes.weakint [46, 39]
48 Ops.ADD dtypes.weakint [47, 31]
49 Ops.INDEX dtypes.f32 [45, 48]
50 Ops.MUL dtypes.f32 [44, 49]
51 Ops.REDUCE dtypes.f32 [50, 42] (Ops.ADD, 0)
52 Ops.CONST dtypes.weakfloat [] -inf
53 Ops.WHERE dtypes.f32 [35, 51, 52]
54 Ops.STORE dtypes.void [33, 53]
55 Ops.END dtypes.void [54, 3, 6, 11, 14, 18, 24, 27, 30]
56 Ops.SINK dtypes.void [55] KernelInfo(name="r_64_32_16_8_4_3_3_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(3, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(3, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), optimized=True)

