# tolk D175 golden=1eb7ce4c40bba3fd37a8316c5fd2995f
0 Ops.CONST dtypes.weakint [] 1000000
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 25
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 5
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 32
11 Ops.RANGE dtypes.weakint [10] (AxisType.WARP, -1)
12 Ops.CONST dtypes.weakint [] 16
13 Ops.DIV dtypes.weakint [11, 12]
14 Ops.MOD dtypes.weakint [13, 8]
15 Ops.ADD dtypes.weakint [9, 14]
16 Ops.MUL dtypes.weakint [15, 8]
17 Ops.CONST dtypes.weakint [] 4
18 Ops.DIV dtypes.weakint [11, 17]
19 Ops.MOD dtypes.weakint [18, 8]
20 Ops.ADD dtypes.weakint [16, 19]
21 Ops.MUL dtypes.weakint [20, 8]
22 Ops.DIV dtypes.weakint [11, 8]
23 Ops.MOD dtypes.weakint [22, 8]
24 Ops.ADD dtypes.weakint [21, 23]
25 Ops.CONST dtypes.weakint [] 1000
26 Ops.MUL dtypes.weakint [24, 25]
27 Ops.CONST dtypes.weakint [] 125
28 Ops.RANGE dtypes.weakint [27] (AxisType.GLOBAL, 2)
29 Ops.MUL dtypes.weakint [28, 8]
30 Ops.CONST dtypes.weakint [] 8
31 Ops.DIV dtypes.weakint [11, 30]
32 Ops.MOD dtypes.weakint [31, 8]
33 Ops.ADD dtypes.weakint [29, 32]
34 Ops.MUL dtypes.weakint [33, 8]
35 Ops.CONST dtypes.weakint [] 1
36 Ops.DIV dtypes.weakint [11, 35]
37 Ops.MOD dtypes.weakint [36, 8]
38 Ops.ADD dtypes.weakint [34, 37]
39 Ops.MUL dtypes.weakint [38, 8]
40 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 3)
41 Ops.ADD dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [26, 41]
43 Ops.INDEX dtypes.f32 [1, 42]
44 Ops.CONST dtypes.weakint [] 16000
45 Ops.PARAM dtypes.f32 [44] ParamArg(slot=1, dtype=dtypes.f32)
46 Ops.MUL dtypes.weakint [24, 12]
47 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
48 Ops.MUL dtypes.weakint [47, 8]
49 Ops.ADD dtypes.weakint [48, 32]
50 Ops.MUL dtypes.weakint [49, 8]
51 Ops.ADD dtypes.weakint [50, 37]
52 Ops.MUL dtypes.weakint [51, 8]
53 Ops.ADD dtypes.weakint [52, 40]
54 Ops.ADD dtypes.weakint [46, 53]
55 Ops.INDEX dtypes.f32 [45, 54]
56 Ops.PARAM dtypes.f32 [44] ParamArg(slot=2, dtype=dtypes.f32)
57 Ops.ADD dtypes.weakint [48, 14]
58 Ops.MUL dtypes.weakint [57, 8]
59 Ops.ADD dtypes.weakint [58, 19]
60 Ops.MUL dtypes.weakint [59, 8]
61 Ops.ADD dtypes.weakint [60, 23]
62 Ops.MUL dtypes.weakint [61, 25]
63 Ops.ADD dtypes.weakint [62, 41]
64 Ops.INDEX dtypes.f32 [56, 63]
65 Ops.CONST dtypes.weakfloat [] 0.0
66 Ops.CAST dtypes.f32 [65] dtypes.f32
67 Ops.STACK dtypes.f32 [66, 66]
68 Ops.WMMA dtypes.f32 [55, 64, 67] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
69 Ops.REDUCE dtypes.f32 [68, 47] (Ops.ADD, 0)
70 Ops.STORE dtypes.void [43, 69]
71 Ops.END dtypes.void [70, 3, 6, 11, 28, 40]
72 Ops.SINK dtypes.void [71] KernelInfo(name="r_25_125_32_2_5_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(5, AxisType.UPCAST))), optimized=True)

