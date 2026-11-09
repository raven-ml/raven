# tolk D89 golden=a402aa598f78e10783a27cb4cca3f8e1
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.CONST dtypes.weakint [] 32
10 Ops.RANGE dtypes.weakint [9] (AxisType.WARP, -1)
11 Ops.DIV dtypes.weakint [10, 2]
12 Ops.MOD dtypes.weakint [11, 4]
13 Ops.ADD dtypes.weakint [8, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.CONST dtypes.weakint [] 8
16 Ops.DIV dtypes.weakint [10, 15]
17 Ops.MOD dtypes.weakint [16, 4]
18 Ops.ADD dtypes.weakint [14, 17]
19 Ops.MUL dtypes.weakint [18, 4]
20 Ops.CONST dtypes.weakint [] 4
21 Ops.DIV dtypes.weakint [10, 20]
22 Ops.MOD dtypes.weakint [21, 4]
23 Ops.ADD dtypes.weakint [19, 22]
24 Ops.CONST dtypes.weakint [] 256
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 9)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.MUL dtypes.weakint [29, 4]
31 Ops.DIV dtypes.weakint [10, 4]
32 Ops.MOD dtypes.weakint [31, 4]
33 Ops.ADD dtypes.weakint [30, 32]
34 Ops.MUL dtypes.weakint [33, 4]
35 Ops.CONST dtypes.weakint [] 1
36 Ops.DIV dtypes.weakint [10, 35]
37 Ops.MOD dtypes.weakint [36, 4]
38 Ops.ADD dtypes.weakint [34, 37]
39 Ops.MUL dtypes.weakint [38, 4]
40 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
41 Ops.ADD dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [25, 41]
43 Ops.INDEX dtypes.f32 [1, 42]
44 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
45 Ops.ADD dtypes.weakint [5, 40]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 12]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 17]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 22]
52 Ops.MUL dtypes.weakint [51, 24]
53 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.ADD dtypes.weakint [54, 6]
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.ADD dtypes.weakint [56, 32]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.ADD dtypes.weakint [58, 37]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
62 Ops.ADD dtypes.weakint [60, 61]
63 Ops.ADD dtypes.weakint [52, 62]
64 Ops.INDEX dtypes.f16 [44, 63]
65 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
66 Ops.ADD dtypes.weakint [60, 40]
67 Ops.MUL dtypes.weakint [66, 24]
68 Ops.ADD dtypes.weakint [30, 12]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.ADD dtypes.weakint [69, 17]
71 Ops.MUL dtypes.weakint [70, 4]
72 Ops.ADD dtypes.weakint [71, 22]
73 Ops.ADD dtypes.weakint [67, 72]
74 Ops.INDEX dtypes.f16 [65, 73]
75 Ops.CONST dtypes.weakfloat [] 0.0
76 Ops.CAST dtypes.f32 [75] dtypes.f32
77 Ops.STACK dtypes.f32 [76, 76, 76, 76]
78 Ops.WMMA dtypes.f32 [64, 74, 77] ((8, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
79 Ops.REDUCE dtypes.f32 [78, 53] (Ops.ADD, 0)
80 Ops.STORE dtypes.void [43, 79]
81 Ops.END dtypes.void [80, 3, 6, 10, 26, 28, 40]
82 Ops.SINK dtypes.void [81] KernelInfo(name="r_16_16_2_32_2_2_2_16", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

