# tolk D280 golden=c46249d6d99ef685128bbb00890836ef
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 12)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 11)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 10)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.MUL dtypes.weakint [22, 4]
24 Ops.CONST dtypes.weakint [] 32
25 Ops.RANGE dtypes.weakint [24] (AxisType.WARP, -1)
26 Ops.CONST dtypes.weakint [] 16
27 Ops.DIV dtypes.weakint [25, 26]
28 Ops.MOD dtypes.weakint [27, 4]
29 Ops.ADD dtypes.weakint [23, 28]
30 Ops.CONST dtypes.weakint [] 8
31 Ops.RANGE dtypes.weakint [30] (AxisType.GLOBAL, 2)
32 Ops.MUL dtypes.weakint [31, 4]
33 Ops.DIV dtypes.weakint [25, 30]
34 Ops.MOD dtypes.weakint [33, 4]
35 Ops.ADD dtypes.weakint [32, 34]
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.CONST dtypes.weakint [] 4
38 Ops.DIV dtypes.weakint [25, 37]
39 Ops.MOD dtypes.weakint [38, 4]
40 Ops.ADD dtypes.weakint [36, 39]
41 Ops.MUL dtypes.weakint [40, 4]
42 Ops.DIV dtypes.weakint [25, 4]
43 Ops.MOD dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [41, 43]
45 Ops.MUL dtypes.weakint [44, 4]
46 Ops.DIV dtypes.weakint [25, 2]
47 Ops.MOD dtypes.weakint [46, 4]
48 Ops.ADD dtypes.weakint [45, 47]
49 Ops.AFTER dtypes.f32 [1, 29, 48]
50 Ops.CONST dtypes.weakint [] 128
51 Ops.MUL dtypes.weakint [29, 50]
52 Ops.ADD dtypes.weakint [51, 48]
53 Ops.INDEX dtypes.f32 [49, 52]
54 Ops.CONST dtypes.weakint [] 2048
55 Ops.PARAM dtypes.f16 [54] ParamArg(slot=1, dtype=dtypes.f16)
56 Ops.ADD dtypes.weakint [14, 34]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.ADD dtypes.weakint [57, 39]
59 Ops.MUL dtypes.weakint [58, 4]
60 Ops.ADD dtypes.weakint [59, 43]
61 Ops.MUL dtypes.weakint [60, 4]
62 Ops.ADD dtypes.weakint [61, 47]
63 Ops.MUL dtypes.weakint [62, 26]
64 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 15]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 18]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.ADD dtypes.weakint [69, 21]
71 Ops.MUL dtypes.weakint [70, 4]
72 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
73 Ops.ADD dtypes.weakint [71, 72]
74 Ops.ADD dtypes.weakint [63, 73]
75 Ops.INDEX dtypes.f16 [55, 74]
76 Ops.PARAM dtypes.f16 [54] ParamArg(slot=2, dtype=dtypes.f16)
77 Ops.MUL dtypes.weakint [73, 50]
78 Ops.ADD dtypes.weakint [77, 48]
79 Ops.INDEX dtypes.f16 [76, 78]
80 Ops.CONST dtypes.weakfloat [] 0.0
81 Ops.CAST dtypes.f32 [80] dtypes.f32
82 Ops.STACK dtypes.f32 [81, 81, 81, 81, 81, 81, 81, 81]
83 Ops.WMMA dtypes.f32 [75, 79, 82] ((16, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
84 Ops.REDUCE dtypes.f32 [83, 64] (Ops.ADD, 0)
85 Ops.STORE dtypes.void [53, 84]
86 Ops.END dtypes.void [85, 3, 6, 9, 12, 15, 18, 21, 25, 31]
87 Ops.SINK dtypes.void [86] KernelInfo(name="r_8_2_2_2_32_2_2_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL))), optimized=True)

