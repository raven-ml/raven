# tolk D280 golden=4f0c0f4e8b4edc229a87738caefeea63
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
50 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
51 Ops.ADD dtypes.weakint [49, 50]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
54 Ops.ADD dtypes.weakint [52, 53]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
57 Ops.ADD dtypes.weakint [55, 56]
58 Ops.CONST dtypes.weakint [] 23
59 Ops.CMPLT dtypes.bool [57, 58]
60 Ops.CONST dtypes.weakint [] 391
61 Ops.PARAM dtypes.f32 [60] ParamArg(slot=1, dtype=dtypes.f32)
62 Ops.AND dtypes.bool [37, 40]
63 Ops.AND dtypes.bool [59, 62]
64 Ops.MUL dtypes.weakint [20, 58]
65 Ops.ADD dtypes.weakint [64, 57]
66 Ops.WHERE dtypes.weakint [63, 65, 45]
67 Ops.INDEX dtypes.f32 [61, 66]
68 Ops.CONST dtypes.weakint [] 667
69 Ops.PARAM dtypes.f32 [68] ParamArg(slot=2, dtype=dtypes.f32)
70 Ops.AND dtypes.bool [59, 41]
71 Ops.MUL dtypes.weakint [57, 38]
72 Ops.ADD dtypes.weakint [71, 34]
73 Ops.WHERE dtypes.weakint [70, 72, 45]
74 Ops.INDEX dtypes.f32 [69, 73]
75 Ops.MUL dtypes.f32 [67, 74]
76 Ops.CONST dtypes.weakfloat [] 0.0
77 Ops.CAST dtypes.f32 [76] dtypes.f32
78 Ops.WHERE dtypes.f32 [59, 75, 77]
79 Ops.REDUCE dtypes.f32 [78, 48, 50, 53, 56] (Ops.ADD, 0)
80 Ops.STORE dtypes.void [47, 79]
81 Ops.END dtypes.void [80, 3, 7, 21, 33]
82 Ops.SINK dtypes.void [81] KernelInfo(name="r_3_4_32_2_3_2_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 2)),), optimized=True)

