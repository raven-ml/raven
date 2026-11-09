# tolk D89 golden=759716ea450e8c3fd7cff73adddfd245
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [2] (AxisType.WARP, -1)
7 Ops.CONST dtypes.weakint [] 16
8 Ops.DIV dtypes.weakint [6, 7]
9 Ops.MOD dtypes.weakint [8, 4]
10 Ops.ADD dtypes.weakint [5, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.DIV dtypes.weakint [6, 12]
14 Ops.MOD dtypes.weakint [13, 4]
15 Ops.ADD dtypes.weakint [11, 14]
16 Ops.MUL dtypes.weakint [15, 4]
17 Ops.DIV dtypes.weakint [6, 4]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [16, 18]
20 Ops.CONST dtypes.weakint [] 256
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.CONST dtypes.weakint [] 8
23 Ops.RANGE dtypes.weakint [22] (AxisType.GLOBAL, 2)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 8)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.MUL dtypes.weakint [29, 4]
31 Ops.DIV dtypes.weakint [6, 22]
32 Ops.MOD dtypes.weakint [31, 4]
33 Ops.ADD dtypes.weakint [30, 32]
34 Ops.MUL dtypes.weakint [33, 4]
35 Ops.CONST dtypes.weakint [] 1
36 Ops.DIV dtypes.weakint [6, 35]
37 Ops.MOD dtypes.weakint [36, 4]
38 Ops.ADD dtypes.weakint [34, 37]
39 Ops.MUL dtypes.weakint [38, 4]
40 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
41 Ops.ADD dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [21, 41]
43 Ops.INDEX dtypes.f32 [1, 42]
44 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
45 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 32]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 37]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 40]
52 Ops.ADD dtypes.weakint [21, 51]
53 Ops.INDEX dtypes.f16 [44, 52]
54 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
55 Ops.ADD dtypes.weakint [46, 9]
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.ADD dtypes.weakint [56, 14]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.ADD dtypes.weakint [58, 18]
60 Ops.MUL dtypes.weakint [59, 20]
61 Ops.ADD dtypes.weakint [60, 41]
62 Ops.INDEX dtypes.f16 [54, 61]
63 Ops.CONST dtypes.weakfloat [] 0.0
64 Ops.CAST dtypes.f32 [63] dtypes.f32
65 Ops.STACK dtypes.f32 [64, 64]
66 Ops.WMMA dtypes.f32 [53, 62, 65] ((8, 8, 8), dtypes.f16, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
67 Ops.REDUCE dtypes.f32 [66, 45] (Ops.ADD, 0)
68 Ops.STORE dtypes.void [43, 67]
69 Ops.END dtypes.void [68, 3, 6, 23, 25, 28, 40]
70 Ops.SINK dtypes.void [69] KernelInfo(name="r_32_8_2_32_2_2_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

