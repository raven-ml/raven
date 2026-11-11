# tolk D175 golden=5c94e982740f217cc0620d17d01399d8
0 Ops.CONST dtypes.weakint [] 49152
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 6
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.RANGE dtypes.weakint [6] (AxisType.WARP, -1)
8 Ops.CONST dtypes.weakint [] 16
9 Ops.DIV dtypes.weakint [7, 8]
10 Ops.MOD dtypes.weakint [9, 4]
11 Ops.ADD dtypes.weakint [5, 10]
12 Ops.MUL dtypes.weakint [11, 4]
13 Ops.CONST dtypes.weakint [] 4
14 Ops.DIV dtypes.weakint [7, 13]
15 Ops.MOD dtypes.weakint [14, 4]
16 Ops.ADD dtypes.weakint [12, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.DIV dtypes.weakint [7, 4]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [17, 19]
21 Ops.CONST dtypes.weakint [] 1024
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.CONST dtypes.weakint [] 64
24 Ops.RANGE dtypes.weakint [23] (AxisType.GLOBAL, 2)
25 Ops.MUL dtypes.weakint [24, 4]
26 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 7)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.CONST dtypes.weakint [] 8
30 Ops.DIV dtypes.weakint [7, 29]
31 Ops.MOD dtypes.weakint [30, 4]
32 Ops.ADD dtypes.weakint [28, 31]
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.CONST dtypes.weakint [] 1
35 Ops.DIV dtypes.weakint [7, 34]
36 Ops.MOD dtypes.weakint [35, 4]
37 Ops.ADD dtypes.weakint [33, 36]
38 Ops.MUL dtypes.weakint [37, 4]
39 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
40 Ops.ADD dtypes.weakint [38, 39]
41 Ops.ADD dtypes.weakint [22, 40]
42 Ops.INDEX dtypes.f32 [1, 41]
43 Ops.CONST dtypes.weakint [] 3072
44 Ops.PARAM dtypes.f32 [43] ParamArg(slot=1, dtype=dtypes.f32)
45 Ops.MUL dtypes.weakint [20, 23]
46 Ops.RANGE dtypes.weakint [29] (AxisType.WEAK, 0)
47 Ops.MUL dtypes.weakint [46, 4]
48 Ops.ADD dtypes.weakint [47, 31]
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 36]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 39]
53 Ops.ADD dtypes.weakint [45, 52]
54 Ops.INDEX dtypes.f32 [44, 53]
55 Ops.CONST dtypes.weakint [] 65536
56 Ops.PARAM dtypes.f32 [55] ParamArg(slot=2, dtype=dtypes.f32)
57 Ops.ADD dtypes.weakint [47, 10]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.ADD dtypes.weakint [58, 15]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 19]
62 Ops.MUL dtypes.weakint [61, 21]
63 Ops.ADD dtypes.weakint [62, 40]
64 Ops.INDEX dtypes.f32 [56, 63]
65 Ops.CONST dtypes.weakfloat [] 0.0
66 Ops.CAST dtypes.f32 [65] dtypes.f32
67 Ops.STACK dtypes.f32 [66, 66]
68 Ops.WMMA dtypes.f32 [54, 64, 67] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
69 Ops.REDUCE dtypes.f32 [68, 46] (Ops.ADD, 0)
70 Ops.STORE dtypes.void [42, 69]
71 Ops.END dtypes.void [70, 3, 7, 24, 26, 39]
72 Ops.SINK dtypes.void [71] KernelInfo(name="r_6_64_2_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

