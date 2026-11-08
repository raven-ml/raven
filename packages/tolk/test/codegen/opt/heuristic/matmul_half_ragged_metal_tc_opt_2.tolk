# tolk D89 D280 golden=6f19a37dee9145884da12efb11744083
0 Ops.CONST dtypes.weakint [] 493
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 3
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
21 Ops.RANGE dtypes.weakint [13] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.CONST dtypes.weakint [] 8
24 Ops.DIV dtypes.weakint [7, 23]
25 Ops.MOD dtypes.weakint [24, 4]
26 Ops.ADD dtypes.weakint [22, 25]
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.CONST dtypes.weakint [] 1
29 Ops.DIV dtypes.weakint [7, 28]
30 Ops.MOD dtypes.weakint [29, 4]
31 Ops.ADD dtypes.weakint [27, 30]
32 Ops.MUL dtypes.weakint [31, 4]
33 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
34 Ops.ADD dtypes.weakint [32, 33]
35 Ops.AFTER dtypes.f32 [1, 20, 34]
36 Ops.CONST dtypes.weakint [] 17
37 Ops.CMPLT dtypes.bool [20, 36]
38 Ops.CONST dtypes.weakint [] 29
39 Ops.CMPLT dtypes.bool [34, 38]
40 Ops.CONST dtypes.bool [] True
41 Ops.AND dtypes.bool [39, 40]
42 Ops.AND dtypes.bool [37, 41]
43 Ops.MUL dtypes.weakint [20, 38]
44 Ops.ADD dtypes.weakint [43, 34]
45 Ops.CONST dtypes.bool [] Invalid
46 Ops.WHERE dtypes.weakint [42, 44, 45]
47 Ops.INDEX dtypes.f32 [35, 46]
48 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 25]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 30]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 33]
55 Ops.CONST dtypes.weakint [] 23
56 Ops.CMPLT dtypes.bool [54, 55]
57 Ops.CONST dtypes.weakint [] 391
58 Ops.PARAM dtypes.f16 [57] ParamArg(slot=1, dtype=dtypes.f16)
59 Ops.AND dtypes.bool [37, 40]
60 Ops.AND dtypes.bool [56, 59]
61 Ops.MUL dtypes.weakint [20, 55]
62 Ops.ADD dtypes.weakint [61, 54]
63 Ops.WHERE dtypes.weakint [60, 62, 45]
64 Ops.INDEX dtypes.f16 [58, 63]
65 Ops.CONST dtypes.weakfloat [] 0.0
66 Ops.CAST dtypes.f16 [65] dtypes.f16
67 Ops.WHERE dtypes.f16 [56, 64, 66]
68 Ops.ADD dtypes.weakint [49, 10]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.ADD dtypes.weakint [69, 15]
71 Ops.MUL dtypes.weakint [70, 4]
72 Ops.ADD dtypes.weakint [71, 19]
73 Ops.CMPLT dtypes.bool [72, 55]
74 Ops.CONST dtypes.weakint [] 667
75 Ops.PARAM dtypes.f16 [74] ParamArg(slot=2, dtype=dtypes.f16)
76 Ops.AND dtypes.bool [73, 41]
77 Ops.MUL dtypes.weakint [72, 38]
78 Ops.ADD dtypes.weakint [77, 34]
79 Ops.WHERE dtypes.weakint [76, 78, 45]
80 Ops.INDEX dtypes.f16 [75, 79]
81 Ops.WHERE dtypes.f16 [73, 80, 66]
82 Ops.CAST dtypes.f32 [65] dtypes.f32
83 Ops.STACK dtypes.f32 [82, 82]
84 Ops.WMMA dtypes.f32 [67, 81, 83] ((8, 8, 8), dtypes.f16, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
85 Ops.REDUCE dtypes.f32 [84, 48] (Ops.ADD, 0)
86 Ops.STORE dtypes.void [47, 85]
87 Ops.END dtypes.void [86, 3, 7, 21, 33]
88 Ops.SINK dtypes.void [87] KernelInfo(name="r_3_4_32_2_3", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1)),), optimized=True)

