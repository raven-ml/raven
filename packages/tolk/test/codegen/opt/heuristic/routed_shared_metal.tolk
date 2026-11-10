# tolk D73 D471
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [2] (AxisType.WARP, -1)
7 Ops.CONST dtypes.weakint [] 16
8 Ops.DIV dtypes.weakint [6, 7]
9 Ops.MOD dtypes.weakint [8, 4]
10 Ops.ADD dtypes.weakint [5, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.DIV dtypes.weakint [6, 12]
14 Ops.MOD dtypes.weakint [13, 4]
15 Ops.ADD dtypes.weakint [11, 14]
16 Ops.MUL dtypes.weakint [15, 4]
17 Ops.DIV dtypes.weakint [6, 4]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [16, 18]
20 Ops.CONST dtypes.weakint [] 8
21 Ops.RANGE dtypes.weakint [20] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.DIV dtypes.weakint [6, 20]
24 Ops.MOD dtypes.weakint [23, 4]
25 Ops.ADD dtypes.weakint [22, 24]
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.CONST dtypes.weakint [] 1
28 Ops.DIV dtypes.weakint [6, 27]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [26, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.AFTER dtypes.f32 [1, 19, 33]
35 Ops.CONST dtypes.weakint [] 64
36 Ops.MUL dtypes.weakint [19, 35]
37 Ops.ADD dtypes.weakint [36, 33]
38 Ops.INDEX dtypes.f32 [34, 37]
39 Ops.CONST dtypes.weakint [] 65536
40 Ops.PARAM dtypes.bf16 [39] ParamArg(slot=1, dtype=dtypes.bf16)
41 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 0, 0)
42 Ops.MUL dtypes.weakint [41, 4]
43 Ops.ADD dtypes.weakint [42, 24]
44 Ops.MUL dtypes.weakint [43, 4]
45 Ops.ADD dtypes.weakint [44, 29]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 32]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.RANGE dtypes.weakint [27] (AxisType.WEAK, 0, 1)
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
52 Ops.ADD dtypes.weakint [50, 51]
53 Ops.ADD dtypes.weakint [48, 52]
54 Ops.CONST dtypes.weakint [] 256
55 Ops.MUL dtypes.weakint [19, 54]
56 Ops.ADD dtypes.weakint [53, 55]
57 Ops.INDEX dtypes.bf16 [40, 56]
58 Ops.CAST dtypes.f32 [57] dtypes.f32
59 Ops.CMPLT dtypes.bool [52, 27]
60 Ops.PARAM dtypes.u8 [0] ParamArg(slot=2, dtype=dtypes.u8)
61 Ops.CONST dtypes.weakint [] 128
62 Ops.MUL dtypes.weakint [33, 61]
63 Ops.ADD dtypes.weakint [42, 9]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 14]
66 Ops.MUL dtypes.weakint [65, 4]
67 Ops.ADD dtypes.weakint [66, 18]
68 Ops.ADD dtypes.weakint [62, 67]
69 Ops.PARAM dtypes.i32 [7] ParamArg(slot=3, dtype=dtypes.i32)
70 Ops.CONST dtypes.weakint [] 0
71 Ops.ADD dtypes.weakint [5, 70]
72 Ops.MUL dtypes.weakint [71, 4]
73 Ops.ADD dtypes.weakint [72, 70]
74 Ops.MUL dtypes.weakint [73, 4]
75 Ops.ADD dtypes.weakint [74, 70]
76 Ops.DIV dtypes.weakint [75, 7]
77 Ops.INDEX dtypes.i32 [69, 76]
78 Ops.MAX dtypes.i32 [77, 70]
79 Ops.CONST dtypes.weakint [] -1
80 Ops.XOR dtypes.i32 [78, 79]
81 Ops.CONST dtypes.weakint [] -2
82 Ops.MAX dtypes.i32 [80, 81]
83 Ops.XOR dtypes.i32 [82, 79]
84 Ops.CAST dtypes.weakint [83] dtypes.weakint
85 Ops.CONST dtypes.weakint [] 8192
86 Ops.MUL dtypes.weakint [84, 85]
87 Ops.ADD dtypes.weakint [68, 86]
88 Ops.INDEX dtypes.u8 [60, 87]
89 Ops.CONST dtypes.weakint [] 15
90 Ops.AND dtypes.u8 [88, 89]
91 Ops.SHR dtypes.u8 [88, 12]
92 Ops.WHERE dtypes.u8 [59, 90, 91]
93 Ops.CAST dtypes.bf16 [92] dtypes.bf16
94 Ops.CONST dtypes.weakint [] 1024
95 Ops.PARAM dtypes.bf16 [94] ParamArg(slot=4, dtype=dtypes.bf16)
96 Ops.MUL dtypes.weakint [33, 20]
97 Ops.DIV dtypes.weakint [67, 7]
98 Ops.ADD dtypes.weakint [96, 97]
99 Ops.CONST dtypes.weakint [] 512
100 Ops.MUL dtypes.weakint [84, 99]
101 Ops.ADD dtypes.weakint [98, 100]
102 Ops.INDEX dtypes.bf16 [95, 101]
103 Ops.MUL dtypes.bf16 [93, 102]
104 Ops.CAST dtypes.f32 [103] dtypes.f32
105 Ops.CONST dtypes.weakfloat [] 0.0
106 Ops.CAST dtypes.f32 [105] dtypes.f32
107 Ops.STACK dtypes.f32 [106, 106]
108 Ops.WMMA dtypes.f32 [58, 104, 107] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
109 Ops.REDUCE dtypes.f32 [108, 41, 49, 51] (Ops.ADD, 0)
110 Ops.STORE dtypes.void [38, 109]
111 Ops.END dtypes.void [110, 3, 6, 21, 32]
112 Ops.SINK dtypes.void [111] KernelInfo(name="r_32_8_32_2_16_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST))), optimized=True)

