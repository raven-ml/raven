# tolk D450 golden=3c5383913061fb4e2efa37cdfa0c46fe
0 Ops.CONST dtypes.weakint [] 32768
1 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 4)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.CONST dtypes.weakint [] 2
6 Ops.RANGE dtypes.weakint [5] (AxisType.GLOBAL, 5)
7 Ops.MUL dtypes.weakint [6, 5]
8 Ops.RANGE dtypes.weakint [5] (AxisType.UPCAST, 8)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.MUL dtypes.weakint [9, 5]
11 Ops.RANGE dtypes.weakint [5] (AxisType.UPCAST, 7)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.MUL dtypes.weakint [12, 5]
14 Ops.RANGE dtypes.weakint [5] (AxisType.UPCAST, 6)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.MUL dtypes.weakint [15, 5]
17 Ops.RANGE dtypes.weakint [2] (AxisType.WARP, -1)
18 Ops.CONST dtypes.weakint [] 16
19 Ops.DIV dtypes.weakint [17, 18]
20 Ops.MOD dtypes.weakint [19, 5]
21 Ops.ADD dtypes.weakint [16, 20]
22 Ops.ADD dtypes.weakint [4, 21]
23 Ops.RANGE dtypes.weakint [5] (AxisType.GLOBAL, 3)
24 Ops.MUL dtypes.weakint [23, 5]
25 Ops.CONST dtypes.weakint [] 8
26 Ops.DIV dtypes.weakint [17, 25]
27 Ops.MOD dtypes.weakint [26, 5]
28 Ops.ADD dtypes.weakint [24, 27]
29 Ops.MUL dtypes.weakint [28, 5]
30 Ops.CONST dtypes.weakint [] 4
31 Ops.DIV dtypes.weakint [17, 30]
32 Ops.MOD dtypes.weakint [31, 5]
33 Ops.ADD dtypes.weakint [29, 32]
34 Ops.MUL dtypes.weakint [33, 5]
35 Ops.DIV dtypes.weakint [17, 5]
36 Ops.MOD dtypes.weakint [35, 5]
37 Ops.ADD dtypes.weakint [34, 36]
38 Ops.MUL dtypes.weakint [37, 5]
39 Ops.CONST dtypes.weakint [] 1
40 Ops.DIV dtypes.weakint [17, 39]
41 Ops.MOD dtypes.weakint [40, 5]
42 Ops.ADD dtypes.weakint [38, 41]
43 Ops.CONST dtypes.weakint [] 1024
44 Ops.MUL dtypes.weakint [42, 43]
45 Ops.ADD dtypes.weakint [22, 44]
46 Ops.INDEX dtypes.f16 [1, 45]
47 Ops.CONST dtypes.weakint [] 3
48 Ops.RANGE dtypes.weakint [47] (AxisType.WEAK, 1)
49 Ops.ADD dtypes.weakint [48, 3]
50 Ops.CMPLT dtypes.bool [49, 39]
51 Ops.CONST dtypes.bool [] False
52 Ops.CMPEQ dtypes.bool [50, 51]
53 Ops.CONST dtypes.weakint [] 33
54 Ops.CMPLT dtypes.bool [49, 53]
55 Ops.AND dtypes.bool [52, 54]
56 Ops.RANGE dtypes.weakint [47] (AxisType.WEAK, 2)
57 Ops.ADD dtypes.weakint [7, 27]
58 Ops.MUL dtypes.weakint [57, 5]
59 Ops.ADD dtypes.weakint [58, 32]
60 Ops.MUL dtypes.weakint [59, 5]
61 Ops.ADD dtypes.weakint [60, 36]
62 Ops.MUL dtypes.weakint [61, 5]
63 Ops.ADD dtypes.weakint [62, 41]
64 Ops.ADD dtypes.weakint [56, 63]
65 Ops.CMPLT dtypes.bool [64, 39]
66 Ops.CMPEQ dtypes.bool [65, 51]
67 Ops.CMPLT dtypes.bool [64, 53]
68 Ops.AND dtypes.bool [66, 67]
69 Ops.AND dtypes.bool [55, 68]
70 Ops.CONST dtypes.weakint [] 16384
71 Ops.PARAM dtypes.f16 [70] ParamArg(slot=1, dtype=dtypes.f16)
72 Ops.AND dtypes.bool [68, 55]
73 Ops.MUL dtypes.weakint [48, 2]
74 Ops.ADD dtypes.weakint [73, 4]
75 Ops.ADD dtypes.weakint [74, 64]
76 Ops.RANGE dtypes.weakint [39] (AxisType.WEAK, 0)
77 Ops.MUL dtypes.weakint [76, 5]
78 Ops.ADD dtypes.weakint [77, 8]
79 Ops.MUL dtypes.weakint [78, 5]
80 Ops.ADD dtypes.weakint [79, 11]
81 Ops.MUL dtypes.weakint [80, 5]
82 Ops.ADD dtypes.weakint [81, 14]
83 Ops.MUL dtypes.weakint [82, 5]
84 Ops.RANGE dtypes.weakint [5] (AxisType.UPCAST, 12)
85 Ops.ADD dtypes.weakint [83, 84]
86 Ops.MUL dtypes.weakint [85, 43]
87 Ops.ADD dtypes.weakint [75, 86]
88 Ops.CONST dtypes.weakint [] -33
89 Ops.ADD dtypes.weakint [87, 88]
90 Ops.CONST dtypes.bool [] Invalid
91 Ops.WHERE dtypes.weakint [72, 89, 90]
92 Ops.INDEX dtypes.f16 [71, 91]
93 Ops.CONST dtypes.weakfloat [] 0.0
94 Ops.WHERE dtypes.f16 [69, 92, 93]
95 Ops.CONST dtypes.weakint [] 4608
96 Ops.PARAM dtypes.f16 [95] ParamArg(slot=2, dtype=dtypes.f16)
97 Ops.MUL dtypes.weakint [48, 47]
98 Ops.ADD dtypes.weakint [97, 56]
99 Ops.CONST dtypes.weakint [] 9
100 Ops.MUL dtypes.weakint [85, 99]
101 Ops.ADD dtypes.weakint [98, 100]
102 Ops.CONST dtypes.weakint [] 144
103 Ops.MUL dtypes.weakint [42, 102]
104 Ops.ADD dtypes.weakint [101, 103]
105 Ops.INDEX dtypes.f16 [96, 104]
106 Ops.CAST dtypes.f32 [93] dtypes.f32
107 Ops.STACK dtypes.f32 [106, 106, 106, 106, 106, 106, 106, 106]
108 Ops.WMMA dtypes.f32 [94, 105, 107] ((16, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 8), 2), ((AxisType.UPCAST, 7), 2), ((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 12), 2)), (((AxisType.UPCAST, 8), 2), ((AxisType.UPCAST, 7), 2), ((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 12), 2)), (((AxisType.UPCAST, 8), 2), ((AxisType.UPCAST, 7), 2), ((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 12), 1))))
109 Ops.REDUCE dtypes.f32 [108, 76, 48, 56] (Ops.ADD, 0)
110 Ops.CAST dtypes.f16 [109] dtypes.f16
111 Ops.STORE dtypes.void [46, 110]
112 Ops.END dtypes.void [111, 23, 17, 3, 6, 8, 11, 14]
113 Ops.SINK dtypes.void [112] KernelInfo(name="r_2_32_2_32_2_2_2_2_3_3", applied_opts=(Opt(op=OptOps.TC, axis=2, arg=(-1, 0, 1)),), optimized=True)

