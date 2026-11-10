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
42 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
43 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0, 0)
44 Ops.MUL dtypes.weakint [43, 4]
45 Ops.ADD dtypes.weakint [44, 6]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 27]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 32]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
52 Ops.ADD dtypes.weakint [50, 51]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.RANGE dtypes.weakint [30] (AxisType.WEAK, 0, 1)
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
57 Ops.ADD dtypes.weakint [55, 56]
58 Ops.ADD dtypes.weakint [53, 57]
59 Ops.ADD dtypes.weakint [5, 35]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 12]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 17]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 22]
66 Ops.MUL dtypes.weakint [65, 38]
67 Ops.ADD dtypes.weakint [58, 66]
68 Ops.INDEX dtypes.bf16 [42, 67]
69 Ops.CMPLT dtypes.bool [57, 30]
70 Ops.CONST dtypes.weakint [] 8192
71 Ops.PARAM dtypes.u8 [70] ParamArg(slot=2, dtype=dtypes.u8)
72 Ops.ADD dtypes.weakint [25, 12]
73 Ops.MUL dtypes.weakint [72, 4]
74 Ops.ADD dtypes.weakint [73, 17]
75 Ops.MUL dtypes.weakint [74, 4]
76 Ops.ADD dtypes.weakint [75, 22]
77 Ops.MUL dtypes.weakint [76, 9]
78 Ops.ADD dtypes.weakint [50, 35]
79 Ops.ADD dtypes.weakint [77, 78]
80 Ops.PARAM dtypes.i32 [20] ParamArg(slot=3, dtype=dtypes.i32)
81 Ops.CONST dtypes.weakint [] 0
82 Ops.ADD dtypes.weakint [5, 81]
83 Ops.MUL dtypes.weakint [82, 4]
84 Ops.ADD dtypes.weakint [83, 81]
85 Ops.MUL dtypes.weakint [84, 4]
86 Ops.ADD dtypes.weakint [85, 81]
87 Ops.MUL dtypes.weakint [86, 4]
88 Ops.ADD dtypes.weakint [87, 81]
89 Ops.DIV dtypes.weakint [88, 38]
90 Ops.INDEX dtypes.i32 [80, 89]
91 Ops.MAX dtypes.i32 [90, 81]
92 Ops.CONST dtypes.weakint [] -1
93 Ops.XOR dtypes.i32 [91, 92]
94 Ops.CONST dtypes.weakint [] -4
95 Ops.MAX dtypes.i32 [93, 94]
96 Ops.XOR dtypes.i32 [95, 92]
97 Ops.CAST dtypes.weakint [96] dtypes.weakint
98 Ops.CONST dtypes.weakint [] 2048
99 Ops.MUL dtypes.weakint [97, 98]
100 Ops.ADD dtypes.weakint [79, 99]
101 Ops.INDEX dtypes.u8 [71, 100]
102 Ops.CONST dtypes.weakint [] 15
103 Ops.AND dtypes.u8 [101, 102]
104 Ops.SHR dtypes.u8 [101, 20]
105 Ops.WHERE dtypes.u8 [69, 103, 104]
106 Ops.CAST dtypes.bf16 [105] dtypes.bf16
107 Ops.CONST dtypes.weakint [] 512
108 Ops.PARAM dtypes.bf16 [107] ParamArg(slot=4, dtype=dtypes.bf16)
109 Ops.MUL dtypes.weakint [76, 4]
110 Ops.DIV dtypes.weakint [78, 2]
111 Ops.ADD dtypes.weakint [109, 110]
112 Ops.CONST dtypes.weakint [] 128
113 Ops.MUL dtypes.weakint [97, 112]
114 Ops.ADD dtypes.weakint [111, 113]
115 Ops.INDEX dtypes.bf16 [108, 114]
116 Ops.MUL dtypes.bf16 [106, 115]
117 Ops.CONST dtypes.weakfloat [] 0.0
118 Ops.CAST dtypes.f32 [117] dtypes.f32
119 Ops.STACK dtypes.f32 [118, 118, 118, 118]
120 Ops.WMMA dtypes.f32 [68, 116, 119] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
121 Ops.REDUCE dtypes.f32 [120, 43, 54, 56] (Ops.ADD, 0)
122 Ops.STORE dtypes.void [41, 121]
123 Ops.END dtypes.void [122, 3, 6, 10, 24, 35]
124 Ops.SINK dtypes.void [123] KernelInfo(name="r_16_8_32_2_2_2_2_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=7, arg=(0, AxisType.UPCAST))), optimized=True)

