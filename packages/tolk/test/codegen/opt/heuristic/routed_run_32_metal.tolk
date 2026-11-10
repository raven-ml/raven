# tolk D73 D471
0 Ops.CONST dtypes.weakint [] 81920
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 160
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.RANGE dtypes.weakint [6] (AxisType.WARP, -1)
8 Ops.CONST dtypes.weakint [] 16
9 Ops.DIV dtypes.weakint [7, 8]
10 Ops.MOD dtypes.weakint [9, 4]
11 Ops.ADD dtypes.weakint [5, 10]
12 Ops.MUL dtypes.weakint [11, 4]
13 Ops.CONST dtypes.weakint [] 4
14 Ops.DIV dtypes.weakint [7, 13]
15 Ops.MOD dtypes.weakint [14, 4]
16 Ops.ADD dtypes.weakint [12, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.DIV dtypes.weakint [7, 4]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [17, 19]
21 Ops.RANGE dtypes.weakint [4] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 13]
23 Ops.RANGE dtypes.weakint [13] (AxisType.LOCAL, 7)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.MUL dtypes.weakint [24, 4]
26 Ops.CONST dtypes.weakint [] 8
27 Ops.DIV dtypes.weakint [7, 26]
28 Ops.MOD dtypes.weakint [27, 4]
29 Ops.ADD dtypes.weakint [25, 28]
30 Ops.MUL dtypes.weakint [29, 4]
31 Ops.CONST dtypes.weakint [] 1
32 Ops.DIV dtypes.weakint [7, 31]
33 Ops.MOD dtypes.weakint [32, 4]
34 Ops.ADD dtypes.weakint [30, 33]
35 Ops.MUL dtypes.weakint [34, 4]
36 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.AFTER dtypes.f32 [1, 20, 37]
39 Ops.CONST dtypes.weakint [] 64
40 Ops.MUL dtypes.weakint [20, 39]
41 Ops.ADD dtypes.weakint [40, 37]
42 Ops.INDEX dtypes.f32 [38, 41]
43 Ops.CONST dtypes.weakint [] 327680
44 Ops.PARAM dtypes.bf16 [43] ParamArg(slot=1, dtype=dtypes.bf16)
45 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0, 0)
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 28]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 33]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 36]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.RANGE dtypes.weakint [31] (AxisType.WEAK, 0, 1)
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
56 Ops.ADD dtypes.weakint [54, 55]
57 Ops.ADD dtypes.weakint [52, 56]
58 Ops.CONST dtypes.weakint [] 256
59 Ops.MUL dtypes.weakint [20, 58]
60 Ops.ADD dtypes.weakint [57, 59]
61 Ops.INDEX dtypes.bf16 [44, 60]
62 Ops.CAST dtypes.f32 [61] dtypes.f32
63 Ops.CMPLT dtypes.bool [56, 31]
64 Ops.CONST dtypes.weakint [] 32768
65 Ops.PARAM dtypes.u8 [64] ParamArg(slot=2, dtype=dtypes.u8)
66 Ops.CONST dtypes.weakint [] 128
67 Ops.MUL dtypes.weakint [37, 66]
68 Ops.ADD dtypes.weakint [46, 10]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.ADD dtypes.weakint [69, 15]
71 Ops.MUL dtypes.weakint [70, 4]
72 Ops.ADD dtypes.weakint [71, 19]
73 Ops.ADD dtypes.weakint [67, 72]
74 Ops.CONST dtypes.weakint [] 40
75 Ops.PARAM dtypes.i32 [74] ParamArg(slot=3, dtype=dtypes.i32)
76 Ops.CONST dtypes.weakint [] 0
77 Ops.ADD dtypes.weakint [5, 76]
78 Ops.MUL dtypes.weakint [77, 4]
79 Ops.ADD dtypes.weakint [78, 76]
80 Ops.MUL dtypes.weakint [79, 4]
81 Ops.ADD dtypes.weakint [80, 76]
82 Ops.DIV dtypes.weakint [81, 6]
83 Ops.INDEX dtypes.i32 [75, 82]
84 Ops.MAX dtypes.i32 [83, 76]
85 Ops.CONST dtypes.weakint [] -1
86 Ops.XOR dtypes.i32 [84, 85]
87 Ops.CONST dtypes.weakint [] -4
88 Ops.MAX dtypes.i32 [86, 87]
89 Ops.XOR dtypes.i32 [88, 85]
90 Ops.CAST dtypes.weakint [89] dtypes.weakint
91 Ops.CONST dtypes.weakint [] 8192
92 Ops.MUL dtypes.weakint [90, 91]
93 Ops.ADD dtypes.weakint [73, 92]
94 Ops.INDEX dtypes.u8 [65, 93]
95 Ops.CONST dtypes.weakint [] 15
96 Ops.AND dtypes.u8 [94, 95]
97 Ops.SHR dtypes.u8 [94, 13]
98 Ops.WHERE dtypes.u8 [63, 96, 97]
99 Ops.CAST dtypes.bf16 [98] dtypes.bf16
100 Ops.CONST dtypes.weakint [] 2048
101 Ops.PARAM dtypes.bf16 [100] ParamArg(slot=4, dtype=dtypes.bf16)
102 Ops.MUL dtypes.weakint [37, 26]
103 Ops.DIV dtypes.weakint [72, 8]
104 Ops.ADD dtypes.weakint [102, 103]
105 Ops.CONST dtypes.weakint [] 512
106 Ops.MUL dtypes.weakint [90, 105]
107 Ops.ADD dtypes.weakint [104, 106]
108 Ops.INDEX dtypes.bf16 [101, 107]
109 Ops.MUL dtypes.bf16 [99, 108]
110 Ops.CAST dtypes.f32 [109] dtypes.f32
111 Ops.CONST dtypes.weakfloat [] 0.0
112 Ops.CAST dtypes.f32 [111] dtypes.f32
113 Ops.STACK dtypes.f32 [112, 112]
114 Ops.WMMA dtypes.f32 [62, 110, 113] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
115 Ops.REDUCE dtypes.f32 [114, 45, 53, 55] (Ops.ADD, 0)
116 Ops.STORE dtypes.void [42, 115]
117 Ops.END dtypes.void [116, 3, 7, 21, 23, 36]
118 Ops.SINK dtypes.void [117] KernelInfo(name="r_160_2_4_32_2_16_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=6, arg=(0, AxisType.UPCAST))), optimized=True)

