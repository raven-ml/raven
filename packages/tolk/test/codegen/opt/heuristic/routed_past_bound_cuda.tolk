# tolk D73 D471
0 Ops.CONST dtypes.weakint [] 10240
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 10
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.CONST dtypes.weakint [] 32
10 Ops.RANGE dtypes.weakint [9] (AxisType.WARP, -1)
11 Ops.CONST dtypes.weakint [] 16
12 Ops.DIV dtypes.weakint [10, 11]
13 Ops.MOD dtypes.weakint [12, 4]
14 Ops.ADD dtypes.weakint [8, 13]
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.CONST dtypes.weakint [] 8
17 Ops.DIV dtypes.weakint [10, 16]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [15, 18]
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.CONST dtypes.weakint [] 4
22 Ops.DIV dtypes.weakint [10, 21]
23 Ops.MOD dtypes.weakint [22, 4]
24 Ops.ADD dtypes.weakint [20, 23]
25 Ops.RANGE dtypes.weakint [16] (AxisType.GLOBAL, 2)
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.DIV dtypes.weakint [10, 4]
28 Ops.MOD dtypes.weakint [27, 4]
29 Ops.ADD dtypes.weakint [26, 28]
30 Ops.MUL dtypes.weakint [29, 4]
31 Ops.CONST dtypes.weakint [] 1
32 Ops.DIV dtypes.weakint [10, 31]
33 Ops.MOD dtypes.weakint [32, 4]
34 Ops.ADD dtypes.weakint [30, 33]
35 Ops.MUL dtypes.weakint [34, 4]
36 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.AFTER dtypes.f32 [1, 24, 37]
39 Ops.CONST dtypes.weakint [] 64
40 Ops.MUL dtypes.weakint [24, 39]
41 Ops.ADD dtypes.weakint [40, 37]
42 Ops.INDEX dtypes.f32 [38, 41]
43 Ops.CONST dtypes.weakint [] 40960
44 Ops.PARAM dtypes.bf16 [43] ParamArg(slot=1, dtype=dtypes.bf16)
45 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 0, 0)
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 6]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 28]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 33]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
54 Ops.ADD dtypes.weakint [52, 53]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.RANGE dtypes.weakint [31] (AxisType.WEAK, 0, 1)
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
59 Ops.ADD dtypes.weakint [57, 58]
60 Ops.ADD dtypes.weakint [55, 59]
61 Ops.ADD dtypes.weakint [5, 36]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 13]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 18]
66 Ops.MUL dtypes.weakint [65, 4]
67 Ops.ADD dtypes.weakint [66, 23]
68 Ops.CONST dtypes.weakint [] 256
69 Ops.MUL dtypes.weakint [67, 68]
70 Ops.ADD dtypes.weakint [60, 69]
71 Ops.INDEX dtypes.bf16 [44, 70]
72 Ops.CMPLT dtypes.bool [59, 31]
73 Ops.CONST dtypes.weakint [] 32768
74 Ops.PARAM dtypes.u8 [73] ParamArg(slot=2, dtype=dtypes.u8)
75 Ops.ADD dtypes.weakint [26, 13]
76 Ops.MUL dtypes.weakint [75, 4]
77 Ops.ADD dtypes.weakint [76, 18]
78 Ops.MUL dtypes.weakint [77, 4]
79 Ops.ADD dtypes.weakint [78, 23]
80 Ops.CONST dtypes.weakint [] 128
81 Ops.MUL dtypes.weakint [79, 80]
82 Ops.ADD dtypes.weakint [52, 36]
83 Ops.ADD dtypes.weakint [81, 82]
84 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
85 Ops.CONST dtypes.weakint [] 0
86 Ops.ADD dtypes.weakint [5, 85]
87 Ops.MUL dtypes.weakint [86, 4]
88 Ops.ADD dtypes.weakint [87, 85]
89 Ops.MUL dtypes.weakint [88, 4]
90 Ops.ADD dtypes.weakint [89, 85]
91 Ops.MUL dtypes.weakint [90, 4]
92 Ops.ADD dtypes.weakint [91, 85]
93 Ops.DIV dtypes.weakint [92, 11]
94 Ops.INDEX dtypes.i32 [84, 93]
95 Ops.MAX dtypes.i32 [94, 85]
96 Ops.CONST dtypes.weakint [] -1
97 Ops.XOR dtypes.i32 [95, 96]
98 Ops.CONST dtypes.weakint [] -4
99 Ops.MAX dtypes.i32 [97, 98]
100 Ops.XOR dtypes.i32 [99, 96]
101 Ops.CAST dtypes.weakint [100] dtypes.weakint
102 Ops.CONST dtypes.weakint [] 8192
103 Ops.MUL dtypes.weakint [101, 102]
104 Ops.ADD dtypes.weakint [83, 103]
105 Ops.INDEX dtypes.u8 [74, 104]
106 Ops.CONST dtypes.weakint [] 15
107 Ops.AND dtypes.u8 [105, 106]
108 Ops.SHR dtypes.u8 [105, 21]
109 Ops.WHERE dtypes.u8 [72, 107, 108]
110 Ops.CAST dtypes.bf16 [109] dtypes.bf16
111 Ops.CONST dtypes.weakint [] 2048
112 Ops.PARAM dtypes.bf16 [111] ParamArg(slot=4, dtype=dtypes.bf16)
113 Ops.MUL dtypes.weakint [79, 16]
114 Ops.DIV dtypes.weakint [82, 11]
115 Ops.ADD dtypes.weakint [113, 114]
116 Ops.CONST dtypes.weakint [] 512
117 Ops.MUL dtypes.weakint [101, 116]
118 Ops.ADD dtypes.weakint [115, 117]
119 Ops.INDEX dtypes.bf16 [112, 118]
120 Ops.MUL dtypes.bf16 [110, 119]
121 Ops.CONST dtypes.weakfloat [] 0.0
122 Ops.CAST dtypes.f32 [121] dtypes.f32
123 Ops.STACK dtypes.f32 [122, 122, 122, 122]
124 Ops.WMMA dtypes.f32 [71, 120, 123] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
125 Ops.REDUCE dtypes.f32 [124, 45, 56, 58] (Ops.ADD, 0)
126 Ops.STORE dtypes.void [42, 125]
127 Ops.END dtypes.void [126, 3, 6, 10, 25, 36]
128 Ops.SINK dtypes.void [127] KernelInfo(name="r_10_8_32_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=7, arg=(0, AxisType.UPCAST))), optimized=True)

