# tolk D89 golden=7b24bc41219480495bbf44478fbaef1c
0 Ops.CONST dtypes.weakint [] 2949120
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 4)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.MUL dtypes.weakint [11, 8]
13 Ops.CONST dtypes.weakint [] 32
14 Ops.RANGE dtypes.weakint [13] (AxisType.WARP, -1)
15 Ops.CONST dtypes.weakint [] 16
16 Ops.DIV dtypes.weakint [14, 15]
17 Ops.MOD dtypes.weakint [16, 8]
18 Ops.ADD dtypes.weakint [12, 17]
19 Ops.MUL dtypes.weakint [18, 8]
20 Ops.DIV dtypes.weakint [14, 2]
21 Ops.MOD dtypes.weakint [20, 8]
22 Ops.ADD dtypes.weakint [19, 21]
23 Ops.MUL dtypes.weakint [22, 8]
24 Ops.DIV dtypes.weakint [14, 4]
25 Ops.MOD dtypes.weakint [24, 8]
26 Ops.ADD dtypes.weakint [23, 25]
27 Ops.CONST dtypes.weakint [] 5760
28 Ops.MUL dtypes.weakint [26, 27]
29 Ops.CONST dtypes.weakint [] 36
30 Ops.RANGE dtypes.weakint [29] (AxisType.GLOBAL, 2)
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 11)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.CONST dtypes.weakint [] 5
35 Ops.MUL dtypes.weakint [33, 34]
36 Ops.RANGE dtypes.weakint [34] (AxisType.UPCAST, 10)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.MUL dtypes.weakint [37, 8]
39 Ops.DIV dtypes.weakint [14, 8]
40 Ops.MOD dtypes.weakint [39, 8]
41 Ops.ADD dtypes.weakint [38, 40]
42 Ops.MUL dtypes.weakint [41, 8]
43 Ops.CONST dtypes.weakint [] 1
44 Ops.DIV dtypes.weakint [14, 43]
45 Ops.MOD dtypes.weakint [44, 8]
46 Ops.ADD dtypes.weakint [42, 45]
47 Ops.MUL dtypes.weakint [46, 8]
48 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 3)
49 Ops.ADD dtypes.weakint [47, 48]
50 Ops.ADD dtypes.weakint [28, 49]
51 Ops.INDEX dtypes.f32 [1, 50]
52 Ops.CONST dtypes.weakint [] 1474560
53 Ops.PARAM dtypes.bf16 [52] ParamArg(slot=1, dtype=dtypes.bf16)
54 Ops.ADD dtypes.weakint [9, 48]
55 Ops.MUL dtypes.weakint [54, 8]
56 Ops.ADD dtypes.weakint [55, 17]
57 Ops.MUL dtypes.weakint [56, 8]
58 Ops.ADD dtypes.weakint [57, 21]
59 Ops.MUL dtypes.weakint [58, 8]
60 Ops.ADD dtypes.weakint [59, 25]
61 Ops.CONST dtypes.weakint [] 2880
62 Ops.MUL dtypes.weakint [60, 61]
63 Ops.CONST dtypes.weakint [] 180
64 Ops.RANGE dtypes.weakint [63] (AxisType.WEAK, 0)
65 Ops.MUL dtypes.weakint [64, 8]
66 Ops.ADD dtypes.weakint [65, 10]
67 Ops.MUL dtypes.weakint [66, 8]
68 Ops.ADD dtypes.weakint [67, 40]
69 Ops.MUL dtypes.weakint [68, 8]
70 Ops.ADD dtypes.weakint [69, 45]
71 Ops.MUL dtypes.weakint [70, 8]
72 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 8)
73 Ops.ADD dtypes.weakint [71, 72]
74 Ops.ADD dtypes.weakint [62, 73]
75 Ops.INDEX dtypes.bf16 [53, 74]
76 Ops.CONST dtypes.weakint [] 16588800
77 Ops.PARAM dtypes.bf16 [76] ParamArg(slot=2, dtype=dtypes.bf16)
78 Ops.ADD dtypes.weakint [71, 48]
79 Ops.MUL dtypes.weakint [78, 27]
80 Ops.ADD dtypes.weakint [38, 17]
81 Ops.MUL dtypes.weakint [80, 8]
82 Ops.ADD dtypes.weakint [81, 21]
83 Ops.MUL dtypes.weakint [82, 8]
84 Ops.ADD dtypes.weakint [83, 25]
85 Ops.ADD dtypes.weakint [79, 84]
86 Ops.INDEX dtypes.bf16 [77, 85]
87 Ops.CONST dtypes.weakfloat [] 0.0
88 Ops.CAST dtypes.f32 [87] dtypes.f32
89 Ops.STACK dtypes.f32 [88, 88, 88, 88]
90 Ops.WMMA dtypes.f32 [75, 86, 89] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
91 Ops.REDUCE dtypes.f32 [90, 64] (Ops.ADD, 0)
92 Ops.STORE dtypes.void [51, 91]
93 Ops.END dtypes.void [92, 3, 6, 10, 14, 30, 32, 36, 48]
94 Ops.SINK dtypes.void [93] KernelInfo(name="r_8_36_4_32_2_2_2_4_5_180", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(5, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)

