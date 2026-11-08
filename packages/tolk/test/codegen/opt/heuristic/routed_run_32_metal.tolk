# tolk D73
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
53 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0, 1)
54 Ops.ADD dtypes.weakint [52, 53]
55 Ops.CONST dtypes.weakint [] 256
56 Ops.MUL dtypes.weakint [20, 55]
57 Ops.ADD dtypes.weakint [54, 56]
58 Ops.INDEX dtypes.bf16 [44, 57]
59 Ops.CAST dtypes.f32 [58] dtypes.f32
60 Ops.CMPLT dtypes.bool [53, 31]
61 Ops.CONST dtypes.weakint [] 32768
62 Ops.PARAM dtypes.u8 [61] ParamArg(slot=2, dtype=dtypes.u8)
63 Ops.CONST dtypes.weakint [] 128
64 Ops.MUL dtypes.weakint [37, 63]
65 Ops.ADD dtypes.weakint [46, 10]
66 Ops.MUL dtypes.weakint [65, 4]
67 Ops.ADD dtypes.weakint [66, 15]
68 Ops.MUL dtypes.weakint [67, 4]
69 Ops.ADD dtypes.weakint [68, 19]
70 Ops.ADD dtypes.weakint [64, 69]
71 Ops.CONST dtypes.weakint [] 40
72 Ops.PARAM dtypes.i32 [71] ParamArg(slot=3, dtype=dtypes.i32)
73 Ops.CONST dtypes.weakint [] 0
74 Ops.ADD dtypes.weakint [5, 73]
75 Ops.MUL dtypes.weakint [74, 4]
76 Ops.ADD dtypes.weakint [75, 73]
77 Ops.MUL dtypes.weakint [76, 4]
78 Ops.ADD dtypes.weakint [77, 73]
79 Ops.DIV dtypes.weakint [78, 6]
80 Ops.INDEX dtypes.i32 [72, 79]
81 Ops.MAX dtypes.i32 [80, 73]
82 Ops.CONST dtypes.weakint [] -1
83 Ops.XOR dtypes.i32 [81, 82]
84 Ops.CONST dtypes.weakint [] -4
85 Ops.MAX dtypes.i32 [83, 84]
86 Ops.XOR dtypes.i32 [85, 82]
87 Ops.CAST dtypes.weakint [86] dtypes.weakint
88 Ops.CONST dtypes.weakint [] 8192
89 Ops.MUL dtypes.weakint [87, 88]
90 Ops.ADD dtypes.weakint [70, 89]
91 Ops.INDEX dtypes.u8 [62, 90]
92 Ops.CONST dtypes.weakint [] 15
93 Ops.AND dtypes.u8 [91, 92]
94 Ops.SHR dtypes.u8 [91, 13]
95 Ops.WHERE dtypes.u8 [60, 93, 94]
96 Ops.CAST dtypes.bf16 [95] dtypes.bf16
97 Ops.CONST dtypes.weakint [] 2048
98 Ops.PARAM dtypes.bf16 [97] ParamArg(slot=4, dtype=dtypes.bf16)
99 Ops.MUL dtypes.weakint [37, 26]
100 Ops.DIV dtypes.weakint [69, 8]
101 Ops.ADD dtypes.weakint [99, 100]
102 Ops.CONST dtypes.weakint [] 512
103 Ops.MUL dtypes.weakint [87, 102]
104 Ops.ADD dtypes.weakint [101, 103]
105 Ops.INDEX dtypes.bf16 [98, 104]
106 Ops.MUL dtypes.bf16 [96, 105]
107 Ops.CAST dtypes.f32 [106] dtypes.f32
108 Ops.CONST dtypes.weakfloat [] 0.0
109 Ops.CAST dtypes.f32 [108] dtypes.f32
110 Ops.STACK dtypes.f32 [109, 109]
111 Ops.WMMA dtypes.f32 [59, 107, 110] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
112 Ops.REDUCE dtypes.f32 [111, 45, 53] (Ops.ADD, 0)
113 Ops.STORE dtypes.void [42, 112]
114 Ops.END dtypes.void [113, 3, 7, 21, 23, 36]
115 Ops.SINK dtypes.void [114] KernelInfo(name="r_160_2_4_32_2_16_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)

