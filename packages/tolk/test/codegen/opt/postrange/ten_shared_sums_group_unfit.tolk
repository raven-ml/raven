# tolk D101 golden=3277fe214448c4a310d5182cabca9a7b
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 1024
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 10)
7 Ops.MUL dtypes.weakint [6, 0]
8 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.INDEX dtypes.f32 [5, 9]
11 Ops.REDUCE dtypes.f32 [10, 6, 8] (Ops.ADD, 0)
12 Ops.PARAM dtypes.f32 [4] ParamArg(slot=2, dtype=dtypes.f32)
13 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
14 Ops.ADD dtypes.weakint [7, 13]
15 Ops.INDEX dtypes.f32 [12, 14]
16 Ops.REDUCE dtypes.f32 [15, 6, 13] (Ops.ADD, 0)
17 Ops.ADD dtypes.f32 [11, 16]
18 Ops.PARAM dtypes.f32 [4] ParamArg(slot=3, dtype=dtypes.f32)
19 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 2)
20 Ops.ADD dtypes.weakint [7, 19]
21 Ops.INDEX dtypes.f32 [18, 20]
22 Ops.REDUCE dtypes.f32 [21, 6, 19] (Ops.ADD, 0)
23 Ops.ADD dtypes.f32 [17, 22]
24 Ops.PARAM dtypes.f32 [4] ParamArg(slot=4, dtype=dtypes.f32)
25 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 3)
26 Ops.ADD dtypes.weakint [7, 25]
27 Ops.INDEX dtypes.f32 [24, 26]
28 Ops.REDUCE dtypes.f32 [27, 6, 25] (Ops.ADD, 0)
29 Ops.ADD dtypes.f32 [23, 28]
30 Ops.PARAM dtypes.f32 [4] ParamArg(slot=5, dtype=dtypes.f32)
31 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 4)
32 Ops.ADD dtypes.weakint [7, 31]
33 Ops.INDEX dtypes.f32 [30, 32]
34 Ops.REDUCE dtypes.f32 [33, 6, 31] (Ops.ADD, 0)
35 Ops.ADD dtypes.f32 [29, 34]
36 Ops.PARAM dtypes.f32 [4] ParamArg(slot=6, dtype=dtypes.f32)
37 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 5)
38 Ops.ADD dtypes.weakint [7, 37]
39 Ops.INDEX dtypes.f32 [36, 38]
40 Ops.REDUCE dtypes.f32 [39, 6, 37] (Ops.ADD, 0)
41 Ops.ADD dtypes.f32 [35, 40]
42 Ops.PARAM dtypes.f32 [4] ParamArg(slot=7, dtype=dtypes.f32)
43 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 6)
44 Ops.ADD dtypes.weakint [7, 43]
45 Ops.INDEX dtypes.f32 [42, 44]
46 Ops.REDUCE dtypes.f32 [45, 6, 43] (Ops.ADD, 0)
47 Ops.ADD dtypes.f32 [41, 46]
48 Ops.PARAM dtypes.f32 [4] ParamArg(slot=8, dtype=dtypes.f32)
49 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 7)
50 Ops.ADD dtypes.weakint [7, 49]
51 Ops.INDEX dtypes.f32 [48, 50]
52 Ops.REDUCE dtypes.f32 [51, 6, 49] (Ops.ADD, 0)
53 Ops.ADD dtypes.f32 [47, 52]
54 Ops.PARAM dtypes.f32 [4] ParamArg(slot=9, dtype=dtypes.f32)
55 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 8)
56 Ops.INDEX dtypes.f32 [54, 55]
57 Ops.REDUCE dtypes.f32 [56, 55] (Ops.ADD, 0)
58 Ops.ADD dtypes.f32 [53, 57]
59 Ops.PARAM dtypes.f32 [4] ParamArg(slot=10, dtype=dtypes.f32)
60 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 9)
61 Ops.INDEX dtypes.f32 [59, 60]
62 Ops.REDUCE dtypes.f32 [61, 60] (Ops.ADD, 0)
63 Ops.ADD dtypes.f32 [58, 62]
64 Ops.STORE dtypes.void [3, 63]
65 Ops.SINK dtypes.void [64] KernelInfo(name="r_1024_1024_1024", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(1024, AxisType.LOCAL, True)),), optimized=True)

