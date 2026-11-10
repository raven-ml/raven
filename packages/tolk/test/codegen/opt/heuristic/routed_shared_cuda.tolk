# tolk D73 D471
0 Ops.CONST dtypes.weakint [] 16384
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
24 Ops.RANGE dtypes.weakint [15] (AxisType.GLOBAL, 2)
25 Ops.MUL dtypes.weakint [24, 4]
26 Ops.DIV dtypes.weakint [10, 4]
27 Ops.MOD dtypes.weakint [26, 4]
28 Ops.ADD dtypes.weakint [25, 27]
29 Ops.MUL dtypes.weakint [28, 4]
30 Ops.CONST dtypes.weakint [] 1
31 Ops.DIV dtypes.weakint [10, 30]
32 Ops.MOD dtypes.weakint [31, 4]
33 Ops.ADD dtypes.weakint [29, 32]
34 Ops.MUL dtypes.weakint [33, 4]
35 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.AFTER dtypes.f32 [1, 23, 36]
38 Ops.CONST dtypes.weakint [] 64
39 Ops.MUL dtypes.weakint [23, 38]
40 Ops.ADD dtypes.weakint [39, 36]
41 Ops.INDEX dtypes.f32 [37, 40]
42 Ops.CONST dtypes.weakint [] 65536
43 Ops.PARAM dtypes.bf16 [42] ParamArg(slot=1, dtype=dtypes.bf16)
44 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0, 0)
45 Ops.MUL dtypes.weakint [44, 4]
46 Ops.ADD dtypes.weakint [45, 6]
47 Ops.MUL dtypes.weakint [46, 4]
48 Ops.ADD dtypes.weakint [47, 27]
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 32]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
53 Ops.ADD dtypes.weakint [51, 52]
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.RANGE dtypes.weakint [30] (AxisType.WEAK, 0, 1)
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
58 Ops.ADD dtypes.weakint [56, 57]
59 Ops.ADD dtypes.weakint [54, 58]
60 Ops.ADD dtypes.weakint [5, 35]
61 Ops.MUL dtypes.weakint [60, 4]
62 Ops.ADD dtypes.weakint [61, 12]
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.ADD dtypes.weakint [63, 17]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 22]
67 Ops.CONST dtypes.weakint [] 256
68 Ops.MUL dtypes.weakint [66, 67]
69 Ops.ADD dtypes.weakint [59, 68]
70 Ops.INDEX dtypes.bf16 [43, 69]
71 Ops.CMPLT dtypes.bool [58, 30]
72 Ops.PARAM dtypes.u8 [0] ParamArg(slot=2, dtype=dtypes.u8)
73 Ops.ADD dtypes.weakint [25, 12]
74 Ops.MUL dtypes.weakint [73, 4]
75 Ops.ADD dtypes.weakint [74, 17]
76 Ops.MUL dtypes.weakint [75, 4]
77 Ops.ADD dtypes.weakint [76, 22]
78 Ops.CONST dtypes.weakint [] 128
79 Ops.MUL dtypes.weakint [77, 78]
80 Ops.ADD dtypes.weakint [51, 35]
81 Ops.ADD dtypes.weakint [79, 80]
82 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
83 Ops.CONST dtypes.weakint [] 0
84 Ops.ADD dtypes.weakint [5, 83]
85 Ops.MUL dtypes.weakint [84, 4]
86 Ops.ADD dtypes.weakint [85, 83]
87 Ops.MUL dtypes.weakint [86, 4]
88 Ops.ADD dtypes.weakint [87, 83]
89 Ops.MUL dtypes.weakint [88, 4]
90 Ops.ADD dtypes.weakint [89, 83]
91 Ops.DIV dtypes.weakint [90, 2]
92 Ops.INDEX dtypes.i32 [82, 91]
93 Ops.MAX dtypes.i32 [92, 83]
94 Ops.CONST dtypes.weakint [] -1
95 Ops.XOR dtypes.i32 [93, 94]
96 Ops.CONST dtypes.weakint [] -2
97 Ops.MAX dtypes.i32 [95, 96]
98 Ops.XOR dtypes.i32 [97, 94]
99 Ops.CAST dtypes.weakint [98] dtypes.weakint
100 Ops.CONST dtypes.weakint [] 8192
101 Ops.MUL dtypes.weakint [99, 100]
102 Ops.ADD dtypes.weakint [81, 101]
103 Ops.INDEX dtypes.u8 [72, 102]
104 Ops.CONST dtypes.weakint [] 15
105 Ops.AND dtypes.u8 [103, 104]
106 Ops.SHR dtypes.u8 [103, 20]
107 Ops.WHERE dtypes.u8 [71, 105, 106]
108 Ops.CAST dtypes.bf16 [107] dtypes.bf16
109 Ops.CONST dtypes.weakint [] 1024
110 Ops.PARAM dtypes.bf16 [109] ParamArg(slot=4, dtype=dtypes.bf16)
111 Ops.MUL dtypes.weakint [77, 15]
112 Ops.DIV dtypes.weakint [80, 2]
113 Ops.ADD dtypes.weakint [111, 112]
114 Ops.CONST dtypes.weakint [] 512
115 Ops.MUL dtypes.weakint [99, 114]
116 Ops.ADD dtypes.weakint [113, 115]
117 Ops.INDEX dtypes.bf16 [110, 116]
118 Ops.MUL dtypes.bf16 [108, 117]
119 Ops.CONST dtypes.weakfloat [] 0.0
120 Ops.CAST dtypes.f32 [119] dtypes.f32
121 Ops.STACK dtypes.f32 [120, 120, 120, 120]
122 Ops.WMMA dtypes.f32 [70, 118, 121] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
123 Ops.REDUCE dtypes.f32 [122, 44, 55, 57] (Ops.ADD, 0)
124 Ops.STORE dtypes.void [41, 123]
125 Ops.END dtypes.void [124, 3, 6, 10, 24, 35]
126 Ops.SINK dtypes.void [125] KernelInfo(name="r_16_8_32_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=7, arg=(0, AxisType.UPCAST))), optimized=True)

