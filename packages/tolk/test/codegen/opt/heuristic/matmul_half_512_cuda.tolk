# tolk D89 golden=ec1e2d2c158ea6999e3005461b5405f9
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
27 Ops.CONST dtypes.weakint [] 512
28 Ops.MUL dtypes.weakint [26, 27]
29 Ops.RANGE dtypes.weakint [13] (AxisType.GLOBAL, 2)
30 Ops.MUL dtypes.weakint [29, 8]
31 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 10)
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.MUL dtypes.weakint [32, 8]
34 Ops.DIV dtypes.weakint [14, 8]
35 Ops.MOD dtypes.weakint [34, 8]
36 Ops.ADD dtypes.weakint [33, 35]
37 Ops.MUL dtypes.weakint [36, 8]
38 Ops.CONST dtypes.weakint [] 1
39 Ops.DIV dtypes.weakint [14, 38]
40 Ops.MOD dtypes.weakint [39, 8]
41 Ops.ADD dtypes.weakint [37, 40]
42 Ops.MUL dtypes.weakint [41, 8]
43 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 3)
44 Ops.ADD dtypes.weakint [42, 43]
45 Ops.ADD dtypes.weakint [28, 44]
46 Ops.INDEX dtypes.f32 [1, 45]
47 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
48 Ops.ADD dtypes.weakint [9, 43]
49 Ops.MUL dtypes.weakint [48, 8]
50 Ops.ADD dtypes.weakint [49, 17]
51 Ops.MUL dtypes.weakint [50, 8]
52 Ops.ADD dtypes.weakint [51, 21]
53 Ops.MUL dtypes.weakint [52, 8]
54 Ops.ADD dtypes.weakint [53, 25]
55 Ops.MUL dtypes.weakint [54, 27]
56 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0)
57 Ops.MUL dtypes.weakint [56, 8]
58 Ops.ADD dtypes.weakint [57, 10]
59 Ops.MUL dtypes.weakint [58, 8]
60 Ops.ADD dtypes.weakint [59, 35]
61 Ops.MUL dtypes.weakint [60, 8]
62 Ops.ADD dtypes.weakint [61, 40]
63 Ops.MUL dtypes.weakint [62, 8]
64 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 8)
65 Ops.ADD dtypes.weakint [63, 64]
66 Ops.ADD dtypes.weakint [55, 65]
67 Ops.INDEX dtypes.f16 [47, 66]
68 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
69 Ops.ADD dtypes.weakint [63, 43]
70 Ops.MUL dtypes.weakint [69, 27]
71 Ops.ADD dtypes.weakint [33, 17]
72 Ops.MUL dtypes.weakint [71, 8]
73 Ops.ADD dtypes.weakint [72, 21]
74 Ops.MUL dtypes.weakint [73, 8]
75 Ops.ADD dtypes.weakint [74, 25]
76 Ops.ADD dtypes.weakint [70, 75]
77 Ops.INDEX dtypes.f16 [68, 76]
78 Ops.CONST dtypes.weakfloat [] 0.0
79 Ops.CAST dtypes.f32 [78] dtypes.f32
80 Ops.STACK dtypes.f32 [79, 79, 79, 79]
81 Ops.WMMA dtypes.f32 [67, 77, 80] ((8, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
82 Ops.REDUCE dtypes.f32 [81, 56] (Ops.ADD, 0)
83 Ops.STORE dtypes.void [46, 82]
84 Ops.END dtypes.void [83, 3, 6, 10, 14, 29, 31, 43]
85 Ops.SINK dtypes.void [84] KernelInfo(name="r_8_32_2_32_2_2_2_4_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

