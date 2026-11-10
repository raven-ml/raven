# tolk D73 D471
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
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
17 Ops.DIV dtypes.weakint [16, 2]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [14, 18]
20 Ops.CONST dtypes.weakint [] 4
21 Ops.RANGE dtypes.weakint [20] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.CONST dtypes.weakint [] 8
24 Ops.DIV dtypes.weakint [16, 23]
25 Ops.MOD dtypes.weakint [24, 4]
26 Ops.ADD dtypes.weakint [22, 25]
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.DIV dtypes.weakint [16, 20]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [27, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.DIV dtypes.weakint [16, 4]
33 Ops.MOD dtypes.weakint [32, 4]
34 Ops.ADD dtypes.weakint [31, 33]
35 Ops.MUL dtypes.weakint [34, 4]
36 Ops.CONST dtypes.weakint [] 1
37 Ops.DIV dtypes.weakint [16, 36]
38 Ops.MOD dtypes.weakint [37, 4]
39 Ops.ADD dtypes.weakint [35, 38]
40 Ops.AFTER dtypes.f32 [1, 19, 39]
41 Ops.CONST dtypes.weakint [] 64
42 Ops.MUL dtypes.weakint [19, 41]
43 Ops.ADD dtypes.weakint [42, 39]
44 Ops.INDEX dtypes.f32 [40, 43]
45 Ops.CONST dtypes.weakint [] 65536
46 Ops.PARAM dtypes.bf16 [45] ParamArg(slot=1, dtype=dtypes.bf16)
47 Ops.RANGE dtypes.weakint [23] (AxisType.WEAK, 0, 0)
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 6]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 9]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.ADD dtypes.weakint [52, 12]
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
56 Ops.ADD dtypes.weakint [54, 55]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.RANGE dtypes.weakint [36] (AxisType.WEAK, 0, 1)
59 Ops.MUL dtypes.weakint [58, 4]
60 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
61 Ops.ADD dtypes.weakint [59, 60]
62 Ops.ADD dtypes.weakint [57, 61]
63 Ops.ADD dtypes.weakint [5, 25]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 29]
66 Ops.MUL dtypes.weakint [65, 4]
67 Ops.ADD dtypes.weakint [66, 33]
68 Ops.MUL dtypes.weakint [67, 4]
69 Ops.ADD dtypes.weakint [68, 38]
70 Ops.CONST dtypes.weakint [] 256
71 Ops.MUL dtypes.weakint [69, 70]
72 Ops.ADD dtypes.weakint [62, 71]
73 Ops.INDEX dtypes.bf16 [46, 72]
74 Ops.CMPLT dtypes.bool [61, 36]
75 Ops.PARAM dtypes.u8 [0] ParamArg(slot=2, dtype=dtypes.u8)
76 Ops.CONST dtypes.weakint [] 128
77 Ops.MUL dtypes.weakint [39, 76]
78 Ops.ADD dtypes.weakint [77, 56]
79 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
80 Ops.CONST dtypes.weakint [] 0
81 Ops.ADD dtypes.weakint [5, 80]
82 Ops.MUL dtypes.weakint [81, 4]
83 Ops.ADD dtypes.weakint [82, 80]
84 Ops.MUL dtypes.weakint [83, 4]
85 Ops.ADD dtypes.weakint [84, 80]
86 Ops.MUL dtypes.weakint [85, 4]
87 Ops.ADD dtypes.weakint [86, 80]
88 Ops.DIV dtypes.weakint [87, 2]
89 Ops.INDEX dtypes.i32 [79, 88]
90 Ops.MAX dtypes.i32 [89, 80]
91 Ops.CONST dtypes.weakint [] -1
92 Ops.XOR dtypes.i32 [90, 91]
93 Ops.CONST dtypes.weakint [] -2
94 Ops.MAX dtypes.i32 [92, 93]
95 Ops.XOR dtypes.i32 [94, 91]
96 Ops.CAST dtypes.weakint [95] dtypes.weakint
97 Ops.CONST dtypes.weakint [] 8192
98 Ops.MUL dtypes.weakint [96, 97]
99 Ops.ADD dtypes.weakint [78, 98]
100 Ops.INDEX dtypes.u8 [75, 99]
101 Ops.CONST dtypes.weakint [] 15
102 Ops.AND dtypes.u8 [100, 101]
103 Ops.SHR dtypes.u8 [100, 20]
104 Ops.WHERE dtypes.u8 [74, 102, 103]
105 Ops.CAST dtypes.bf16 [104] dtypes.bf16
106 Ops.CONST dtypes.weakint [] 1024
107 Ops.PARAM dtypes.bf16 [106] ParamArg(slot=4, dtype=dtypes.bf16)
108 Ops.MUL dtypes.weakint [39, 23]
109 Ops.DIV dtypes.weakint [56, 2]
110 Ops.ADD dtypes.weakint [108, 109]
111 Ops.CONST dtypes.weakint [] 512
112 Ops.MUL dtypes.weakint [96, 111]
113 Ops.ADD dtypes.weakint [110, 112]
114 Ops.INDEX dtypes.bf16 [107, 113]
115 Ops.MUL dtypes.bf16 [105, 114]
116 Ops.CONST dtypes.weakfloat [] 0.0
117 Ops.CAST dtypes.f32 [116] dtypes.f32
118 Ops.STACK dtypes.f32 [117, 117, 117, 117, 117, 117, 117, 117]
119 Ops.WMMA dtypes.f32 [73, 115, 118] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
120 Ops.REDUCE dtypes.f32 [119, 47, 58, 60] (Ops.ADD, 0)
121 Ops.STORE dtypes.void [44, 120]
122 Ops.END dtypes.void [121, 3, 6, 9, 12, 16, 21]
123 Ops.SINK dtypes.void [122] KernelInfo(name="r_16_4_32_2_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=8, arg=(0, AxisType.UPCAST))), optimized=True)

