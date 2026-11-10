# tolk D73 D471
0 Ops.CONST dtypes.weakint [] 81920
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 80
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
25 Ops.RANGE dtypes.weakint [21] (AxisType.GLOBAL, 2)
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 9)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.MUL dtypes.weakint [28, 4]
30 Ops.DIV dtypes.weakint [10, 4]
31 Ops.MOD dtypes.weakint [30, 4]
32 Ops.ADD dtypes.weakint [29, 31]
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.CONST dtypes.weakint [] 1
35 Ops.DIV dtypes.weakint [10, 34]
36 Ops.MOD dtypes.weakint [35, 4]
37 Ops.ADD dtypes.weakint [33, 36]
38 Ops.MUL dtypes.weakint [37, 4]
39 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
40 Ops.ADD dtypes.weakint [38, 39]
41 Ops.AFTER dtypes.f32 [1, 24, 40]
42 Ops.CONST dtypes.weakint [] 64
43 Ops.MUL dtypes.weakint [24, 42]
44 Ops.ADD dtypes.weakint [43, 40]
45 Ops.INDEX dtypes.f32 [41, 44]
46 Ops.CONST dtypes.weakint [] 327680
47 Ops.PARAM dtypes.bf16 [46] ParamArg(slot=1, dtype=dtypes.bf16)
48 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 0, 0)
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 6]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 31]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 36]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
57 Ops.ADD dtypes.weakint [55, 56]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.RANGE dtypes.weakint [34] (AxisType.WEAK, 0, 1)
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
62 Ops.ADD dtypes.weakint [60, 61]
63 Ops.ADD dtypes.weakint [58, 62]
64 Ops.ADD dtypes.weakint [5, 39]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 13]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 18]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.ADD dtypes.weakint [69, 23]
71 Ops.CONST dtypes.weakint [] 256
72 Ops.MUL dtypes.weakint [70, 71]
73 Ops.ADD dtypes.weakint [63, 72]
74 Ops.INDEX dtypes.bf16 [47, 73]
75 Ops.CMPLT dtypes.bool [62, 34]
76 Ops.CONST dtypes.weakint [] 32768
77 Ops.PARAM dtypes.u8 [76] ParamArg(slot=2, dtype=dtypes.u8)
78 Ops.ADD dtypes.weakint [29, 13]
79 Ops.MUL dtypes.weakint [78, 4]
80 Ops.ADD dtypes.weakint [79, 18]
81 Ops.MUL dtypes.weakint [80, 4]
82 Ops.ADD dtypes.weakint [81, 23]
83 Ops.CONST dtypes.weakint [] 128
84 Ops.MUL dtypes.weakint [82, 83]
85 Ops.ADD dtypes.weakint [55, 39]
86 Ops.ADD dtypes.weakint [84, 85]
87 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
88 Ops.CONST dtypes.weakint [] 0
89 Ops.ADD dtypes.weakint [5, 88]
90 Ops.MUL dtypes.weakint [89, 4]
91 Ops.ADD dtypes.weakint [90, 88]
92 Ops.MUL dtypes.weakint [91, 4]
93 Ops.ADD dtypes.weakint [92, 88]
94 Ops.MUL dtypes.weakint [93, 4]
95 Ops.ADD dtypes.weakint [94, 88]
96 Ops.DIV dtypes.weakint [95, 11]
97 Ops.INDEX dtypes.i32 [87, 96]
98 Ops.MAX dtypes.i32 [97, 88]
99 Ops.CONST dtypes.weakint [] -1
100 Ops.XOR dtypes.i32 [98, 99]
101 Ops.CONST dtypes.weakint [] -4
102 Ops.MAX dtypes.i32 [100, 101]
103 Ops.XOR dtypes.i32 [102, 99]
104 Ops.CAST dtypes.weakint [103] dtypes.weakint
105 Ops.CONST dtypes.weakint [] 8192
106 Ops.MUL dtypes.weakint [104, 105]
107 Ops.ADD dtypes.weakint [86, 106]
108 Ops.INDEX dtypes.u8 [77, 107]
109 Ops.CONST dtypes.weakint [] 15
110 Ops.AND dtypes.u8 [108, 109]
111 Ops.SHR dtypes.u8 [108, 21]
112 Ops.WHERE dtypes.u8 [75, 110, 111]
113 Ops.CAST dtypes.bf16 [112] dtypes.bf16
114 Ops.CONST dtypes.weakint [] 2048
115 Ops.PARAM dtypes.bf16 [114] ParamArg(slot=4, dtype=dtypes.bf16)
116 Ops.MUL dtypes.weakint [82, 16]
117 Ops.DIV dtypes.weakint [85, 11]
118 Ops.ADD dtypes.weakint [116, 117]
119 Ops.CONST dtypes.weakint [] 512
120 Ops.MUL dtypes.weakint [104, 119]
121 Ops.ADD dtypes.weakint [118, 120]
122 Ops.INDEX dtypes.bf16 [115, 121]
123 Ops.MUL dtypes.bf16 [113, 122]
124 Ops.CONST dtypes.weakfloat [] 0.0
125 Ops.CAST dtypes.f32 [124] dtypes.f32
126 Ops.STACK dtypes.f32 [125, 125, 125, 125]
127 Ops.WMMA dtypes.f32 [74, 123, 126] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
128 Ops.REDUCE dtypes.f32 [127, 48, 59, 61] (Ops.ADD, 0)
129 Ops.STORE dtypes.void [45, 128]
130 Ops.END dtypes.void [129, 3, 6, 10, 25, 27, 39]
131 Ops.SINK dtypes.void [130] KernelInfo(name="r_80_4_2_32_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=8, arg=(0, AxisType.UPCAST))), optimized=True)

