# tolk D89 D280 golden=f996b1fa1c5ae0ef8e7149a1852310f0
0 Ops.CONST dtypes.weakint [] 493
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 4)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.MUL dtypes.weakint [6, 2]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.RANGE dtypes.weakint [8] (AxisType.WARP, -1)
10 Ops.CONST dtypes.weakint [] 16
11 Ops.DIV dtypes.weakint [9, 10]
12 Ops.MOD dtypes.weakint [11, 2]
13 Ops.ADD dtypes.weakint [7, 12]
14 Ops.MUL dtypes.weakint [13, 2]
15 Ops.CONST dtypes.weakint [] 8
16 Ops.DIV dtypes.weakint [9, 15]
17 Ops.MOD dtypes.weakint [16, 2]
18 Ops.ADD dtypes.weakint [14, 17]
19 Ops.MUL dtypes.weakint [18, 2]
20 Ops.CONST dtypes.weakint [] 4
21 Ops.DIV dtypes.weakint [9, 20]
22 Ops.MOD dtypes.weakint [21, 2]
23 Ops.ADD dtypes.weakint [19, 22]
24 Ops.RANGE dtypes.weakint [20] (AxisType.GLOBAL, 2)
25 Ops.MUL dtypes.weakint [24, 2]
26 Ops.DIV dtypes.weakint [9, 2]
27 Ops.MOD dtypes.weakint [26, 2]
28 Ops.ADD dtypes.weakint [25, 27]
29 Ops.MUL dtypes.weakint [28, 2]
30 Ops.CONST dtypes.weakint [] 1
31 Ops.DIV dtypes.weakint [9, 30]
32 Ops.MOD dtypes.weakint [31, 2]
33 Ops.ADD dtypes.weakint [29, 32]
34 Ops.MUL dtypes.weakint [33, 2]
35 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.AFTER dtypes.f32 [1, 23, 36]
38 Ops.CONST dtypes.weakint [] 17
39 Ops.CMPLT dtypes.bool [23, 38]
40 Ops.CONST dtypes.weakint [] 29
41 Ops.CMPLT dtypes.bool [36, 40]
42 Ops.CONST dtypes.bool [] True
43 Ops.AND dtypes.bool [41, 42]
44 Ops.AND dtypes.bool [39, 43]
45 Ops.MUL dtypes.weakint [23, 40]
46 Ops.ADD dtypes.weakint [45, 36]
47 Ops.CONST dtypes.bool [] Invalid
48 Ops.WHERE dtypes.weakint [44, 46, 47]
49 Ops.INDEX dtypes.f32 [37, 48]
50 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
51 Ops.MUL dtypes.weakint [50, 2]
52 Ops.ADD dtypes.weakint [51, 5]
53 Ops.MUL dtypes.weakint [52, 2]
54 Ops.ADD dtypes.weakint [53, 27]
55 Ops.MUL dtypes.weakint [54, 2]
56 Ops.ADD dtypes.weakint [55, 32]
57 Ops.MUL dtypes.weakint [56, 2]
58 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 8)
59 Ops.ADD dtypes.weakint [57, 58]
60 Ops.CONST dtypes.weakint [] 23
61 Ops.CMPLT dtypes.bool [59, 60]
62 Ops.CONST dtypes.weakint [] 391
63 Ops.PARAM dtypes.f16 [62] ParamArg(slot=1, dtype=dtypes.f16)
64 Ops.ADD dtypes.weakint [4, 35]
65 Ops.MUL dtypes.weakint [64, 2]
66 Ops.ADD dtypes.weakint [65, 12]
67 Ops.MUL dtypes.weakint [66, 2]
68 Ops.ADD dtypes.weakint [67, 17]
69 Ops.MUL dtypes.weakint [68, 2]
70 Ops.ADD dtypes.weakint [69, 22]
71 Ops.CMPLT dtypes.bool [70, 38]
72 Ops.AND dtypes.bool [71, 42]
73 Ops.AND dtypes.bool [61, 72]
74 Ops.MUL dtypes.weakint [70, 60]
75 Ops.ADD dtypes.weakint [74, 59]
76 Ops.WHERE dtypes.weakint [73, 75, 47]
77 Ops.INDEX dtypes.f16 [63, 76]
78 Ops.CONST dtypes.weakfloat [] 0.0
79 Ops.CAST dtypes.f16 [78] dtypes.f16
80 Ops.WHERE dtypes.f16 [61, 77, 79]
81 Ops.ADD dtypes.weakint [57, 35]
82 Ops.CMPLT dtypes.bool [81, 60]
83 Ops.CONST dtypes.weakint [] 667
84 Ops.PARAM dtypes.f16 [83] ParamArg(slot=2, dtype=dtypes.f16)
85 Ops.ADD dtypes.weakint [25, 12]
86 Ops.MUL dtypes.weakint [85, 2]
87 Ops.ADD dtypes.weakint [86, 17]
88 Ops.MUL dtypes.weakint [87, 2]
89 Ops.ADD dtypes.weakint [88, 22]
90 Ops.CMPLT dtypes.bool [89, 40]
91 Ops.AND dtypes.bool [90, 42]
92 Ops.AND dtypes.bool [82, 91]
93 Ops.MUL dtypes.weakint [81, 40]
94 Ops.ADD dtypes.weakint [93, 89]
95 Ops.WHERE dtypes.weakint [92, 94, 47]
96 Ops.INDEX dtypes.f16 [84, 95]
97 Ops.WHERE dtypes.f16 [82, 96, 79]
98 Ops.CAST dtypes.f32 [78] dtypes.f32
99 Ops.STACK dtypes.f32 [98, 98, 98, 98]
100 Ops.WMMA dtypes.f32 [80, 97, 99] ((8, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
101 Ops.REDUCE dtypes.f32 [100, 50] (Ops.ADD, 0)
102 Ops.STORE dtypes.void [49, 101]
103 Ops.END dtypes.void [102, 3, 5, 9, 24, 35]
104 Ops.SINK dtypes.void [103] KernelInfo(name="r_2_4_32_2_2_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1)),), optimized=True)

