# tolk D73 D471
0 Ops.CONST dtypes.weakint [] 81920
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 80
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.CONST dtypes.weakint [] 32
16 Ops.RANGE dtypes.weakint [15] (AxisType.WARP, -1)
17 Ops.CONST dtypes.weakint [] 16
18 Ops.DIV dtypes.weakint [16, 17]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [14, 19]
21 Ops.CONST dtypes.weakint [] 4
22 Ops.RANGE dtypes.weakint [21] (AxisType.GLOBAL, 2)
23 Ops.MUL dtypes.weakint [22, 4]
24 Ops.CONST dtypes.weakint [] 8
25 Ops.DIV dtypes.weakint [16, 24]
26 Ops.MOD dtypes.weakint [25, 4]
27 Ops.ADD dtypes.weakint [23, 26]
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.DIV dtypes.weakint [16, 21]
30 Ops.MOD dtypes.weakint [29, 4]
31 Ops.ADD dtypes.weakint [28, 30]
32 Ops.MUL dtypes.weakint [31, 4]
33 Ops.DIV dtypes.weakint [16, 4]
34 Ops.MOD dtypes.weakint [33, 4]
35 Ops.ADD dtypes.weakint [32, 34]
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.CONST dtypes.weakint [] 1
38 Ops.DIV dtypes.weakint [16, 37]
39 Ops.MOD dtypes.weakint [38, 4]
40 Ops.ADD dtypes.weakint [36, 39]
41 Ops.AFTER dtypes.f32 [1, 20, 40]
42 Ops.CONST dtypes.weakint [] 64
43 Ops.MUL dtypes.weakint [20, 42]
44 Ops.ADD dtypes.weakint [43, 40]
45 Ops.INDEX dtypes.f32 [41, 44]
46 Ops.CONST dtypes.weakint [] 327680
47 Ops.PARAM dtypes.bf16 [46] ParamArg(slot=1, dtype=dtypes.bf16)
48 Ops.RANGE dtypes.weakint [24] (AxisType.WEAK, 0, 0)
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 6]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 9]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 12]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
57 Ops.ADD dtypes.weakint [55, 56]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.RANGE dtypes.weakint [37] (AxisType.WEAK, 0, 1)
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
62 Ops.ADD dtypes.weakint [60, 61]
63 Ops.ADD dtypes.weakint [58, 62]
64 Ops.ADD dtypes.weakint [5, 26]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 30]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 34]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.ADD dtypes.weakint [69, 39]
71 Ops.CONST dtypes.weakint [] 256
72 Ops.MUL dtypes.weakint [70, 71]
73 Ops.ADD dtypes.weakint [63, 72]
74 Ops.INDEX dtypes.bf16 [47, 73]
75 Ops.CMPLT dtypes.bool [62, 37]
76 Ops.CONST dtypes.weakint [] 32768
77 Ops.PARAM dtypes.u8 [76] ParamArg(slot=2, dtype=dtypes.u8)
78 Ops.CONST dtypes.weakint [] 128
79 Ops.MUL dtypes.weakint [40, 78]
80 Ops.ADD dtypes.weakint [79, 57]
81 Ops.CONST dtypes.weakint [] 20
82 Ops.PARAM dtypes.i32 [81] ParamArg(slot=3, dtype=dtypes.i32)
83 Ops.CONST dtypes.weakint [] 0
84 Ops.ADD dtypes.weakint [5, 83]
85 Ops.MUL dtypes.weakint [84, 4]
86 Ops.ADD dtypes.weakint [85, 83]
87 Ops.MUL dtypes.weakint [86, 4]
88 Ops.ADD dtypes.weakint [87, 83]
89 Ops.MUL dtypes.weakint [88, 4]
90 Ops.ADD dtypes.weakint [89, 83]
91 Ops.DIV dtypes.weakint [90, 42]
92 Ops.INDEX dtypes.i32 [82, 91]
93 Ops.MAX dtypes.i32 [92, 83]
94 Ops.CONST dtypes.weakint [] -1
95 Ops.XOR dtypes.i32 [93, 94]
96 Ops.CONST dtypes.weakint [] -4
97 Ops.MAX dtypes.i32 [95, 96]
98 Ops.XOR dtypes.i32 [97, 94]
99 Ops.CAST dtypes.weakint [98] dtypes.weakint
100 Ops.CONST dtypes.weakint [] 8192
101 Ops.MUL dtypes.weakint [99, 100]
102 Ops.ADD dtypes.weakint [80, 101]
103 Ops.INDEX dtypes.u8 [77, 102]
104 Ops.CONST dtypes.weakint [] 15
105 Ops.AND dtypes.u8 [103, 104]
106 Ops.SHR dtypes.u8 [103, 21]
107 Ops.WHERE dtypes.u8 [75, 105, 106]
108 Ops.CAST dtypes.bf16 [107] dtypes.bf16
109 Ops.CONST dtypes.weakint [] 2048
110 Ops.PARAM dtypes.bf16 [109] ParamArg(slot=4, dtype=dtypes.bf16)
111 Ops.MUL dtypes.weakint [40, 24]
112 Ops.DIV dtypes.weakint [57, 17]
113 Ops.ADD dtypes.weakint [111, 112]
114 Ops.CONST dtypes.weakint [] 512
115 Ops.MUL dtypes.weakint [99, 114]
116 Ops.ADD dtypes.weakint [113, 115]
117 Ops.INDEX dtypes.bf16 [110, 116]
118 Ops.MUL dtypes.bf16 [108, 117]
119 Ops.CONST dtypes.weakfloat [] 0.0
120 Ops.CAST dtypes.f32 [119] dtypes.f32
121 Ops.STACK dtypes.f32 [120, 120, 120, 120, 120, 120, 120, 120]
122 Ops.WMMA dtypes.f32 [74, 118, 121] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
123 Ops.REDUCE dtypes.f32 [122, 48, 59, 61] (Ops.ADD, 0)
124 Ops.STORE dtypes.void [45, 123]
125 Ops.END dtypes.void [124, 3, 6, 9, 12, 16, 22]
126 Ops.SINK dtypes.void [125] KernelInfo(name="r_80_4_32_2_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=8, arg=(0, AxisType.UPCAST))), optimized=True)

