# tolk D73
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
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
21 Ops.DIV dtypes.weakint [10, 2]
22 Ops.MOD dtypes.weakint [21, 4]
23 Ops.ADD dtypes.weakint [20, 22]
24 Ops.RANGE dtypes.weakint [16] (AxisType.GLOBAL, 2)
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
42 Ops.CONST dtypes.weakint [] 16384
43 Ops.PARAM dtypes.bf16 [42] ParamArg(slot=1, dtype=dtypes.bf16)
44 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 0, 0)
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
62 Ops.ADD dtypes.weakint [61, 13]
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.ADD dtypes.weakint [63, 18]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 22]
67 Ops.CONST dtypes.weakint [] 256
68 Ops.MUL dtypes.weakint [66, 67]
69 Ops.ADD dtypes.weakint [59, 68]
70 Ops.INDEX dtypes.bf16 [43, 69]
71 Ops.CMPLT dtypes.bool [58, 30]
72 Ops.CONST dtypes.weakint [] 32768
73 Ops.PARAM dtypes.u8 [72] ParamArg(slot=2, dtype=dtypes.u8)
74 Ops.ADD dtypes.weakint [25, 13]
75 Ops.MUL dtypes.weakint [74, 4]
76 Ops.ADD dtypes.weakint [75, 18]
77 Ops.MUL dtypes.weakint [76, 4]
78 Ops.ADD dtypes.weakint [77, 22]
79 Ops.CONST dtypes.weakint [] 128
80 Ops.MUL dtypes.weakint [78, 79]
81 Ops.ADD dtypes.weakint [51, 35]
82 Ops.ADD dtypes.weakint [80, 81]
83 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
84 Ops.CONST dtypes.weakint [] 0
85 Ops.ADD dtypes.weakint [5, 84]
86 Ops.MUL dtypes.weakint [85, 4]
87 Ops.ADD dtypes.weakint [86, 84]
88 Ops.MUL dtypes.weakint [87, 4]
89 Ops.ADD dtypes.weakint [88, 84]
90 Ops.MUL dtypes.weakint [89, 4]
91 Ops.ADD dtypes.weakint [90, 84]
92 Ops.DIV dtypes.weakint [91, 11]
93 Ops.INDEX dtypes.i32 [83, 92]
94 Ops.MAX dtypes.i32 [93, 84]
95 Ops.CONST dtypes.weakint [] -1
96 Ops.XOR dtypes.i32 [94, 95]
97 Ops.CONST dtypes.weakint [] -4
98 Ops.MAX dtypes.i32 [96, 97]
99 Ops.XOR dtypes.i32 [98, 95]
100 Ops.CAST dtypes.weakint [99] dtypes.weakint
101 Ops.CONST dtypes.weakint [] 8192
102 Ops.MUL dtypes.weakint [100, 101]
103 Ops.ADD dtypes.weakint [82, 102]
104 Ops.INDEX dtypes.u8 [73, 103]
105 Ops.CONST dtypes.weakint [] 15
106 Ops.AND dtypes.u8 [104, 105]
107 Ops.SHR dtypes.u8 [104, 2]
108 Ops.WHERE dtypes.u8 [71, 106, 107]
109 Ops.CAST dtypes.bf16 [108] dtypes.bf16
110 Ops.CONST dtypes.weakint [] 2048
111 Ops.PARAM dtypes.bf16 [110] ParamArg(slot=4, dtype=dtypes.bf16)
112 Ops.MUL dtypes.weakint [78, 16]
113 Ops.DIV dtypes.weakint [81, 11]
114 Ops.ADD dtypes.weakint [112, 113]
115 Ops.CONST dtypes.weakint [] 512
116 Ops.MUL dtypes.weakint [100, 115]
117 Ops.ADD dtypes.weakint [114, 116]
118 Ops.INDEX dtypes.bf16 [111, 117]
119 Ops.MUL dtypes.bf16 [109, 118]
120 Ops.CONST dtypes.weakfloat [] 0.0
121 Ops.CAST dtypes.f32 [120] dtypes.f32
122 Ops.STACK dtypes.f32 [121, 121, 121, 121]
123 Ops.WMMA dtypes.f32 [70, 119, 122] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
124 Ops.REDUCE dtypes.f32 [123, 44, 55, 57] (Ops.ADD, 0)
125 Ops.STORE dtypes.void [41, 124]
126 Ops.END dtypes.void [125, 3, 6, 10, 24, 35]
127 Ops.SINK dtypes.void [126] KernelInfo(name="r_4_8_32_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=7, arg=(0, AxisType.UPCAST))), optimized=True)

