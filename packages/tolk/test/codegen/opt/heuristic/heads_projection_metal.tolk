# tolk D175 D491 golden=456a64b54adcc55760f045f24fe95fb3
0 Ops.CONST dtypes.weakint [] 1802240
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 4)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.CONST dtypes.weakint [] 32
10 Ops.RANGE dtypes.weakint [9] (AxisType.WARP, -1)
11 Ops.CONST dtypes.weakint [] 8
12 Ops.DIV dtypes.weakint [10, 11]
13 Ops.MOD dtypes.weakint [12, 4]
14 Ops.ADD dtypes.weakint [8, 13]
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.CONST dtypes.weakint [] 1
17 Ops.DIV dtypes.weakint [10, 16]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [15, 18]
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.CONST dtypes.weakint [] 110
24 Ops.CMPLT dtypes.bool [22, 23]
25 Ops.CONST dtypes.bool [] True
26 Ops.AND dtypes.bool [24, 25]
27 Ops.RANGE dtypes.weakint [11] (AxisType.GLOBAL, 3)
28 Ops.CONST dtypes.weakint [] 4
29 Ops.MUL dtypes.weakint [27, 28]
30 Ops.RANGE dtypes.weakint [28] (AxisType.UPCAST, 9)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.MUL dtypes.weakint [31, 4]
33 Ops.CONST dtypes.weakint [] 16
34 Ops.DIV dtypes.weakint [10, 33]
35 Ops.MOD dtypes.weakint [34, 4]
36 Ops.ADD dtypes.weakint [32, 35]
37 Ops.MUL dtypes.weakint [36, 4]
38 Ops.DIV dtypes.weakint [10, 28]
39 Ops.MOD dtypes.weakint [38, 4]
40 Ops.ADD dtypes.weakint [37, 39]
41 Ops.MUL dtypes.weakint [40, 4]
42 Ops.DIV dtypes.weakint [10, 4]
43 Ops.MOD dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [41, 43]
45 Ops.MUL dtypes.weakint [44, 23]
46 Ops.ADD dtypes.weakint [45, 22]
47 Ops.RANGE dtypes.weakint [28] (AxisType.GLOBAL, 2)
48 Ops.CONST dtypes.weakint [] 28160
49 Ops.MUL dtypes.weakint [47, 48]
50 Ops.ADD dtypes.weakint [46, 49]
51 Ops.RANGE dtypes.weakint [33] (AxisType.GLOBAL, 1)
52 Ops.CONST dtypes.weakint [] 112640
53 Ops.MUL dtypes.weakint [51, 52]
54 Ops.ADD dtypes.weakint [50, 53]
55 Ops.CONST dtypes.bool [] Invalid
56 Ops.WHERE dtypes.weakint [26, 54, 55]
57 Ops.INDEX dtypes.f32 [1, 56]
58 Ops.CONST dtypes.weakint [] 262144
59 Ops.PARAM dtypes.f32 [58] ParamArg(slot=1, dtype=dtypes.f32)
60 Ops.CONST dtypes.weakint [] 64
61 Ops.MUL dtypes.weakint [44, 60]
62 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.ADD dtypes.weakint [63, 13]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 18]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 21]
69 Ops.ADD dtypes.weakint [61, 68]
70 Ops.CONST dtypes.weakint [] 16384
71 Ops.MUL dtypes.weakint [51, 70]
72 Ops.ADD dtypes.weakint [69, 71]
73 Ops.INDEX dtypes.f32 [59, 72]
74 Ops.PARAM dtypes.f32 [48] ParamArg(slot=2, dtype=dtypes.f32)
75 Ops.MUL dtypes.weakint [47, 23]
76 Ops.ADD dtypes.weakint [75, 22]
77 Ops.ADD dtypes.weakint [63, 35]
78 Ops.MUL dtypes.weakint [77, 4]
79 Ops.ADD dtypes.weakint [78, 39]
80 Ops.MUL dtypes.weakint [79, 4]
81 Ops.ADD dtypes.weakint [80, 43]
82 Ops.CONST dtypes.weakint [] 440
83 Ops.MUL dtypes.weakint [81, 82]
84 Ops.ADD dtypes.weakint [76, 83]
85 Ops.WHERE dtypes.weakint [26, 84, 55]
86 Ops.INDEX dtypes.f32 [74, 85]
87 Ops.CONST dtypes.weakfloat [] 0.0
88 Ops.CAST dtypes.f32 [87] dtypes.f32
89 Ops.STACK dtypes.f32 [88, 88]
90 Ops.WMMA dtypes.f32 [73, 86, 89] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 5), 2),), (((AxisType.UPCAST, 5), 2),), (((AxisType.UPCAST, 5), 2),)))
91 Ops.REDUCE dtypes.f32 [90, 62] (Ops.ADD, 0)
92 Ops.STORE dtypes.void [57, 91]
93 Ops.END dtypes.void [92, 51, 47, 27, 30, 10, 3, 6, 21]
94 Ops.SINK dtypes.void [93] KernelInfo(name="r_16_4_8_7_32_2_4_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(2, AxisType.UPCAST))), optimized=True)

