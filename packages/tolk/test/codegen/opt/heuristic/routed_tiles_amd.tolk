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
45 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
46 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0, 0)
47 Ops.MUL dtypes.weakint [46, 4]
48 Ops.ADD dtypes.weakint [47, 6]
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 9]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 12]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
55 Ops.ADD dtypes.weakint [53, 54]
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.RANGE dtypes.weakint [36] (AxisType.WEAK, 0, 1)
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
60 Ops.ADD dtypes.weakint [58, 59]
61 Ops.ADD dtypes.weakint [56, 60]
62 Ops.ADD dtypes.weakint [5, 25]
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.ADD dtypes.weakint [63, 29]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 33]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 38]
69 Ops.MUL dtypes.weakint [68, 41]
70 Ops.ADD dtypes.weakint [61, 69]
71 Ops.INDEX dtypes.bf16 [45, 70]
72 Ops.CMPLT dtypes.bool [60, 36]
73 Ops.CONST dtypes.weakint [] 8192
74 Ops.PARAM dtypes.u8 [73] ParamArg(slot=2, dtype=dtypes.u8)
75 Ops.MUL dtypes.weakint [39, 15]
76 Ops.ADD dtypes.weakint [75, 55]
77 Ops.PARAM dtypes.i32 [20] ParamArg(slot=3, dtype=dtypes.i32)
78 Ops.CONST dtypes.weakint [] 0
79 Ops.ADD dtypes.weakint [5, 78]
80 Ops.MUL dtypes.weakint [79, 4]
81 Ops.ADD dtypes.weakint [80, 78]
82 Ops.MUL dtypes.weakint [81, 4]
83 Ops.ADD dtypes.weakint [82, 78]
84 Ops.MUL dtypes.weakint [83, 4]
85 Ops.ADD dtypes.weakint [84, 78]
86 Ops.DIV dtypes.weakint [85, 41]
87 Ops.INDEX dtypes.i32 [77, 86]
88 Ops.MAX dtypes.i32 [87, 78]
89 Ops.CONST dtypes.weakint [] -1
90 Ops.XOR dtypes.i32 [88, 89]
91 Ops.CONST dtypes.weakint [] -4
92 Ops.MAX dtypes.i32 [90, 91]
93 Ops.XOR dtypes.i32 [92, 89]
94 Ops.CAST dtypes.weakint [93] dtypes.weakint
95 Ops.CONST dtypes.weakint [] 2048
96 Ops.MUL dtypes.weakint [94, 95]
97 Ops.ADD dtypes.weakint [76, 96]
98 Ops.INDEX dtypes.u8 [74, 97]
99 Ops.CONST dtypes.weakint [] 15
100 Ops.AND dtypes.u8 [98, 99]
101 Ops.SHR dtypes.u8 [98, 20]
102 Ops.WHERE dtypes.u8 [72, 100, 101]
103 Ops.CAST dtypes.bf16 [102] dtypes.bf16
104 Ops.CONST dtypes.weakint [] 512
105 Ops.PARAM dtypes.bf16 [104] ParamArg(slot=4, dtype=dtypes.bf16)
106 Ops.MUL dtypes.weakint [39, 4]
107 Ops.DIV dtypes.weakint [55, 2]
108 Ops.ADD dtypes.weakint [106, 107]
109 Ops.CONST dtypes.weakint [] 128
110 Ops.MUL dtypes.weakint [94, 109]
111 Ops.ADD dtypes.weakint [108, 110]
112 Ops.INDEX dtypes.bf16 [105, 111]
113 Ops.MUL dtypes.bf16 [103, 112]
114 Ops.CONST dtypes.weakfloat [] 0.0
115 Ops.CAST dtypes.f32 [114] dtypes.f32
116 Ops.STACK dtypes.f32 [115, 115, 115, 115, 115, 115, 115, 115]
117 Ops.WMMA dtypes.f32 [71, 113, 116] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
118 Ops.REDUCE dtypes.f32 [117, 46, 57, 59] (Ops.ADD, 0)
119 Ops.STORE dtypes.void [44, 118]
120 Ops.END dtypes.void [119, 3, 6, 9, 12, 16, 21]
121 Ops.SINK dtypes.void [120] KernelInfo(name="r_16_4_32_2_2_2_2_2_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=8, arg=(0, AxisType.UPCAST))), optimized=True)

