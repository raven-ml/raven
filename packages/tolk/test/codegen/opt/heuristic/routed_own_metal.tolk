# tolk D73
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
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
21 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.DIV dtypes.weakint [7, 2]
24 Ops.MOD dtypes.weakint [23, 4]
25 Ops.ADD dtypes.weakint [22, 24]
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.CONST dtypes.weakint [] 1
28 Ops.DIV dtypes.weakint [7, 27]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [26, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.AFTER dtypes.f32 [1, 20, 33]
35 Ops.CONST dtypes.weakint [] 64
36 Ops.MUL dtypes.weakint [20, 35]
37 Ops.ADD dtypes.weakint [36, 33]
38 Ops.INDEX dtypes.f32 [34, 37]
39 Ops.CONST dtypes.weakint [] 16384
40 Ops.PARAM dtypes.bf16 [39] ParamArg(slot=1, dtype=dtypes.bf16)
41 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0, 0)
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
55 Ops.MUL dtypes.weakint [20, 54]
56 Ops.ADD dtypes.weakint [53, 55]
57 Ops.INDEX dtypes.bf16 [40, 56]
58 Ops.CAST dtypes.f32 [57] dtypes.f32
59 Ops.CMPLT dtypes.bool [52, 27]
60 Ops.CONST dtypes.weakint [] 32768
61 Ops.PARAM dtypes.u8 [60] ParamArg(slot=2, dtype=dtypes.u8)
62 Ops.CONST dtypes.weakint [] 128
63 Ops.MUL dtypes.weakint [33, 62]
64 Ops.ADD dtypes.weakint [42, 10]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 15]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 19]
69 Ops.ADD dtypes.weakint [63, 68]
70 Ops.PARAM dtypes.i32 [13] ParamArg(slot=3, dtype=dtypes.i32)
71 Ops.CONST dtypes.weakint [] 0
72 Ops.ADD dtypes.weakint [5, 71]
73 Ops.MUL dtypes.weakint [72, 4]
74 Ops.ADD dtypes.weakint [73, 71]
75 Ops.MUL dtypes.weakint [74, 4]
76 Ops.ADD dtypes.weakint [75, 71]
77 Ops.DIV dtypes.weakint [76, 8]
78 Ops.INDEX dtypes.i32 [70, 77]
79 Ops.MAX dtypes.i32 [78, 71]
80 Ops.CONST dtypes.weakint [] -1
81 Ops.XOR dtypes.i32 [79, 80]
82 Ops.CONST dtypes.weakint [] -4
83 Ops.MAX dtypes.i32 [81, 82]
84 Ops.XOR dtypes.i32 [83, 80]
85 Ops.CAST dtypes.weakint [84] dtypes.weakint
86 Ops.CONST dtypes.weakint [] 8192
87 Ops.MUL dtypes.weakint [85, 86]
88 Ops.ADD dtypes.weakint [69, 87]
89 Ops.INDEX dtypes.u8 [61, 88]
90 Ops.CONST dtypes.weakint [] 15
91 Ops.AND dtypes.u8 [89, 90]
92 Ops.SHR dtypes.u8 [89, 13]
93 Ops.WHERE dtypes.u8 [59, 91, 92]
94 Ops.CAST dtypes.bf16 [93] dtypes.bf16
95 Ops.CONST dtypes.weakint [] 2048
96 Ops.PARAM dtypes.bf16 [95] ParamArg(slot=4, dtype=dtypes.bf16)
97 Ops.MUL dtypes.weakint [33, 2]
98 Ops.DIV dtypes.weakint [68, 8]
99 Ops.ADD dtypes.weakint [97, 98]
100 Ops.CONST dtypes.weakint [] 512
101 Ops.MUL dtypes.weakint [85, 100]
102 Ops.ADD dtypes.weakint [99, 101]
103 Ops.INDEX dtypes.bf16 [96, 102]
104 Ops.MUL dtypes.bf16 [94, 103]
105 Ops.CAST dtypes.f32 [104] dtypes.f32
106 Ops.CONST dtypes.weakfloat [] 0.0
107 Ops.CAST dtypes.f32 [106] dtypes.f32
108 Ops.STACK dtypes.f32 [107, 107]
109 Ops.WMMA dtypes.f32 [58, 105, 108] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
110 Ops.REDUCE dtypes.f32 [109, 41, 49, 51] (Ops.ADD, 0)
111 Ops.STORE dtypes.void [38, 110]
112 Ops.END dtypes.void [111, 3, 7, 21, 32]
113 Ops.SINK dtypes.void [112] KernelInfo(name="r_8_8_32_2_16_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST))), optimized=True)

