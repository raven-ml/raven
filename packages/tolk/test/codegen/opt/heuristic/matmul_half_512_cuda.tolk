# tolk D89 D280 golden=ec1e2d2c158ea6999e3005461b5405f9
0 Ops.CONST dtypes.weakint [] 262144
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
27 Ops.RANGE dtypes.weakint [13] (AxisType.GLOBAL, 2)
28 Ops.MUL dtypes.weakint [27, 8]
29 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 10)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.MUL dtypes.weakint [30, 8]
32 Ops.DIV dtypes.weakint [14, 8]
33 Ops.MOD dtypes.weakint [32, 8]
34 Ops.ADD dtypes.weakint [31, 33]
35 Ops.MUL dtypes.weakint [34, 8]
36 Ops.CONST dtypes.weakint [] 1
37 Ops.DIV dtypes.weakint [14, 36]
38 Ops.MOD dtypes.weakint [37, 8]
39 Ops.ADD dtypes.weakint [35, 38]
40 Ops.MUL dtypes.weakint [39, 8]
41 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 3)
42 Ops.ADD dtypes.weakint [40, 41]
43 Ops.AFTER dtypes.f32 [1, 26, 42]
44 Ops.CONST dtypes.weakint [] 512
45 Ops.MUL dtypes.weakint [26, 44]
46 Ops.ADD dtypes.weakint [45, 42]
47 Ops.INDEX dtypes.f32 [43, 46]
48 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
49 Ops.ADD dtypes.weakint [9, 41]
50 Ops.MUL dtypes.weakint [49, 8]
51 Ops.ADD dtypes.weakint [50, 17]
52 Ops.MUL dtypes.weakint [51, 8]
53 Ops.ADD dtypes.weakint [52, 21]
54 Ops.MUL dtypes.weakint [53, 8]
55 Ops.ADD dtypes.weakint [54, 25]
56 Ops.MUL dtypes.weakint [55, 44]
57 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0)
58 Ops.MUL dtypes.weakint [57, 8]
59 Ops.ADD dtypes.weakint [58, 10]
60 Ops.MUL dtypes.weakint [59, 8]
61 Ops.ADD dtypes.weakint [60, 33]
62 Ops.MUL dtypes.weakint [61, 8]
63 Ops.ADD dtypes.weakint [62, 38]
64 Ops.MUL dtypes.weakint [63, 8]
65 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 8)
66 Ops.ADD dtypes.weakint [64, 65]
67 Ops.ADD dtypes.weakint [56, 66]
68 Ops.INDEX dtypes.f16 [48, 67]
69 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
70 Ops.ADD dtypes.weakint [64, 41]
71 Ops.MUL dtypes.weakint [70, 44]
72 Ops.ADD dtypes.weakint [31, 17]
73 Ops.MUL dtypes.weakint [72, 8]
74 Ops.ADD dtypes.weakint [73, 21]
75 Ops.MUL dtypes.weakint [74, 8]
76 Ops.ADD dtypes.weakint [75, 25]
77 Ops.ADD dtypes.weakint [71, 76]
78 Ops.INDEX dtypes.f16 [69, 77]
79 Ops.CONST dtypes.weakfloat [] 0.0
80 Ops.CAST dtypes.f32 [79] dtypes.f32
81 Ops.STACK dtypes.f32 [80, 80, 80, 80]
82 Ops.WMMA dtypes.f32 [68, 78, 81] ((8, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
83 Ops.REDUCE dtypes.f32 [82, 57] (Ops.ADD, 0)
84 Ops.STORE dtypes.void [47, 83]
85 Ops.END dtypes.void [84, 3, 6, 10, 14, 27, 29, 41]
86 Ops.SINK dtypes.void [85] KernelInfo(name="r_8_32_2_32_2_2_2_4_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

