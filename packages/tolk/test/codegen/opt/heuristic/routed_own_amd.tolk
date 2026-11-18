# tolk D73
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
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
21 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.CONST dtypes.weakint [] 8
24 Ops.DIV dtypes.weakint [16, 23]
25 Ops.MOD dtypes.weakint [24, 4]
26 Ops.ADD dtypes.weakint [22, 25]
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.DIV dtypes.weakint [16, 2]
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
40 Ops.AFTER dtypes.f32 [1, 20, 39]
41 Ops.CONST dtypes.weakint [] 64
42 Ops.MUL dtypes.weakint [20, 41]
43 Ops.ADD dtypes.weakint [42, 39]
44 Ops.INDEX dtypes.f32 [40, 43]
45 Ops.CONST dtypes.weakint [] 16384
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
75 Ops.CONST dtypes.weakint [] 32768
76 Ops.PARAM dtypes.u8 [75] ParamArg(slot=2, dtype=dtypes.u8)
77 Ops.CONST dtypes.weakint [] 128
78 Ops.MUL dtypes.weakint [39, 77]
79 Ops.ADD dtypes.weakint [78, 56]
80 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
81 Ops.CONST dtypes.weakint [] 0
82 Ops.ADD dtypes.weakint [5, 81]
83 Ops.MUL dtypes.weakint [82, 4]
84 Ops.ADD dtypes.weakint [83, 81]
85 Ops.MUL dtypes.weakint [84, 4]
86 Ops.ADD dtypes.weakint [85, 81]
87 Ops.MUL dtypes.weakint [86, 4]
88 Ops.ADD dtypes.weakint [87, 81]
89 Ops.DIV dtypes.weakint [88, 17]
90 Ops.INDEX dtypes.i32 [80, 89]
91 Ops.MAX dtypes.i32 [90, 81]
92 Ops.CONST dtypes.weakint [] -1
93 Ops.XOR dtypes.i32 [91, 92]
94 Ops.CONST dtypes.weakint [] -4
95 Ops.MAX dtypes.i32 [93, 94]
96 Ops.XOR dtypes.i32 [95, 92]
97 Ops.CAST dtypes.weakint [96] dtypes.weakint
98 Ops.CONST dtypes.weakint [] 8192
99 Ops.MUL dtypes.weakint [97, 98]
100 Ops.ADD dtypes.weakint [79, 99]
101 Ops.INDEX dtypes.u8 [76, 100]
102 Ops.CONST dtypes.weakint [] 15
103 Ops.AND dtypes.u8 [101, 102]
104 Ops.SHR dtypes.u8 [101, 2]
105 Ops.WHERE dtypes.u8 [74, 103, 104]
106 Ops.CAST dtypes.bf16 [105] dtypes.bf16
107 Ops.CONST dtypes.weakint [] 2048
108 Ops.PARAM dtypes.bf16 [107] ParamArg(slot=4, dtype=dtypes.bf16)
109 Ops.MUL dtypes.weakint [39, 23]
110 Ops.DIV dtypes.weakint [56, 17]
111 Ops.ADD dtypes.weakint [109, 110]
112 Ops.CONST dtypes.weakint [] 512
113 Ops.MUL dtypes.weakint [97, 112]
114 Ops.ADD dtypes.weakint [111, 113]
115 Ops.INDEX dtypes.bf16 [108, 114]
116 Ops.MUL dtypes.bf16 [106, 115]
117 Ops.CONST dtypes.weakfloat [] 0.0
118 Ops.CAST dtypes.f32 [117] dtypes.f32
119 Ops.STACK dtypes.f32 [118, 118, 118, 118, 118, 118, 118, 118]
120 Ops.WMMA dtypes.f32 [73, 116, 119] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
121 Ops.REDUCE dtypes.f32 [120, 47, 58, 60] (Ops.ADD, 0)
122 Ops.STORE dtypes.void [44, 121]
123 Ops.END dtypes.void [122, 3, 6, 9, 12, 16, 21]
124 Ops.SINK dtypes.void [123] KernelInfo(name="r_4_4_32_2_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=8, arg=(0, AxisType.UPCAST))), optimized=True)

