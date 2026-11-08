# tolk D73
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
57 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0, 1)
58 Ops.ADD dtypes.weakint [56, 57]
59 Ops.ADD dtypes.weakint [5, 25]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 29]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 33]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 38]
66 Ops.MUL dtypes.weakint [65, 41]
67 Ops.ADD dtypes.weakint [58, 66]
68 Ops.INDEX dtypes.bf16 [45, 67]
69 Ops.CMPLT dtypes.bool [57, 36]
70 Ops.CONST dtypes.weakint [] 8192
71 Ops.PARAM dtypes.u8 [70] ParamArg(slot=2, dtype=dtypes.u8)
72 Ops.MUL dtypes.weakint [39, 15]
73 Ops.ADD dtypes.weakint [72, 55]
74 Ops.PARAM dtypes.i32 [20] ParamArg(slot=3, dtype=dtypes.i32)
75 Ops.CONST dtypes.weakint [] 0
76 Ops.ADD dtypes.weakint [5, 75]
77 Ops.MUL dtypes.weakint [76, 4]
78 Ops.ADD dtypes.weakint [77, 75]
79 Ops.MUL dtypes.weakint [78, 4]
80 Ops.ADD dtypes.weakint [79, 75]
81 Ops.MUL dtypes.weakint [80, 4]
82 Ops.ADD dtypes.weakint [81, 75]
83 Ops.DIV dtypes.weakint [82, 41]
84 Ops.INDEX dtypes.i32 [74, 83]
85 Ops.MAX dtypes.i32 [84, 75]
86 Ops.CONST dtypes.weakint [] -1
87 Ops.XOR dtypes.i32 [85, 86]
88 Ops.CONST dtypes.weakint [] -4
89 Ops.MAX dtypes.i32 [87, 88]
90 Ops.XOR dtypes.i32 [89, 86]
91 Ops.CAST dtypes.weakint [90] dtypes.weakint
92 Ops.CONST dtypes.weakint [] 2048
93 Ops.MUL dtypes.weakint [91, 92]
94 Ops.ADD dtypes.weakint [73, 93]
95 Ops.INDEX dtypes.u8 [71, 94]
96 Ops.CONST dtypes.weakint [] 15
97 Ops.AND dtypes.u8 [95, 96]
98 Ops.SHR dtypes.u8 [95, 20]
99 Ops.WHERE dtypes.u8 [69, 97, 98]
100 Ops.CAST dtypes.bf16 [99] dtypes.bf16
101 Ops.CONST dtypes.weakint [] 512
102 Ops.PARAM dtypes.bf16 [101] ParamArg(slot=4, dtype=dtypes.bf16)
103 Ops.MUL dtypes.weakint [39, 4]
104 Ops.DIV dtypes.weakint [55, 2]
105 Ops.ADD dtypes.weakint [103, 104]
106 Ops.CONST dtypes.weakint [] 128
107 Ops.MUL dtypes.weakint [91, 106]
108 Ops.ADD dtypes.weakint [105, 107]
109 Ops.INDEX dtypes.bf16 [102, 108]
110 Ops.MUL dtypes.bf16 [100, 109]
111 Ops.CONST dtypes.weakfloat [] 0.0
112 Ops.CAST dtypes.f32 [111] dtypes.f32
113 Ops.STACK dtypes.f32 [112, 112, 112, 112, 112, 112, 112, 112]
114 Ops.WMMA dtypes.f32 [68, 110, 113] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
115 Ops.REDUCE dtypes.f32 [114, 46, 57] (Ops.ADD, 0)
116 Ops.STORE dtypes.void [44, 115]
117 Ops.END dtypes.void [116, 3, 6, 9, 12, 16, 21]
118 Ops.SINK dtypes.void [117] KernelInfo(name="r_16_4_32_2_2_2_2_2_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)),), optimized=True)

