# tolk D101 golden=d9070e1e1ece3901dfbbf4e2b662f63f
0 Ops.CONST dtypes.weakint [] 7
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 4)
4 Ops.MUL dtypes.weakint [3, 0]
5 Ops.RANGE dtypes.weakint [0] (AxisType.LOCAL, 5)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.INDEX dtypes.f32 [1, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 7)
10 Ops.CONST dtypes.weakint [] 13
11 Ops.CMPLT dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 1001
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.CONST dtypes.bool [] True
15 Ops.AND dtypes.bool [11, 14]
16 Ops.CONST dtypes.weakint [] 143
17 Ops.MUL dtypes.weakint [6, 16]
18 Ops.MUL dtypes.weakint [9, 2]
19 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.CONST dtypes.weakint [] 11
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 6)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.ADD dtypes.weakint [17, 24]
26 Ops.CONST dtypes.bool [] Invalid
27 Ops.WHERE dtypes.weakint [15, 25, 26]
28 Ops.INDEX dtypes.f32 [13, 27]
29 Ops.CONST dtypes.weakfloat [] 0.0
30 Ops.CAST dtypes.f32 [29] dtypes.f32
31 Ops.WHERE dtypes.f32 [11, 28, 30]
32 Ops.REDUCE dtypes.f32 [31, 9, 19, 23] (Ops.ADD, 0)
33 Ops.MUL dtypes.weakint [9, 21]
34 Ops.RANGE dtypes.weakint [21] (AxisType.WEAK, 2)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.ADD dtypes.weakint [17, 35]
37 Ops.WHERE dtypes.weakint [15, 36, 26]
38 Ops.INDEX dtypes.f32 [13, 37]
39 Ops.CONST dtypes.weakfloat [] -inf
40 Ops.CAST dtypes.f32 [39] dtypes.f32
41 Ops.WHERE dtypes.f32 [11, 38, 40]
42 Ops.REDUCE dtypes.f32 [41, 9, 34] (Ops.MAX, 0)
43 Ops.ADD dtypes.f32 [32, 42]
44 Ops.STORE dtypes.void [7, 43]
45 Ops.END dtypes.void [44, 3, 5]
46 Ops.SINK dtypes.void [45] KernelInfo(name="r_7_32_11_11", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(11, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.LOCAL, True)), Opt(op=OptOps.PADTO, axis=1, arg=32)), optimized=True)

