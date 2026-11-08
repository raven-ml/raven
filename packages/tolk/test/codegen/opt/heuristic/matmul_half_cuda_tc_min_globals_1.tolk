# tolk D280 golden=c28d7afa2660319331c783e34b959ddb
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 2]
9 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 4)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 2]
12 Ops.CONST dtypes.weakint [] 32
13 Ops.RANGE dtypes.weakint [12] (AxisType.WARP, -1)
14 Ops.CONST dtypes.weakint [] 16
15 Ops.DIV dtypes.weakint [13, 14]
16 Ops.MOD dtypes.weakint [15, 2]
17 Ops.ADD dtypes.weakint [11, 16]
18 Ops.MUL dtypes.weakint [17, 2]
19 Ops.CONST dtypes.weakint [] 8
20 Ops.DIV dtypes.weakint [13, 19]
21 Ops.MOD dtypes.weakint [20, 2]
22 Ops.ADD dtypes.weakint [18, 21]
23 Ops.MUL dtypes.weakint [22, 2]
24 Ops.DIV dtypes.weakint [13, 4]
25 Ops.MOD dtypes.weakint [24, 2]
26 Ops.ADD dtypes.weakint [23, 25]
27 Ops.CONST dtypes.weakint [] 1
28 Ops.RANGE dtypes.weakint [27] (AxisType.GLOBAL, 2)
29 Ops.MUL dtypes.weakint [28, 4]
30 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 11)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.MUL dtypes.weakint [31, 4]
33 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 10)
34 Ops.ADD dtypes.weakint [32, 33]
35 Ops.MUL dtypes.weakint [34, 2]
36 Ops.DIV dtypes.weakint [13, 2]
37 Ops.MOD dtypes.weakint [36, 2]
38 Ops.ADD dtypes.weakint [35, 37]
39 Ops.MUL dtypes.weakint [38, 2]
40 Ops.DIV dtypes.weakint [13, 27]
41 Ops.MOD dtypes.weakint [40, 2]
42 Ops.ADD dtypes.weakint [39, 41]
43 Ops.MUL dtypes.weakint [42, 2]
44 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
45 Ops.ADD dtypes.weakint [43, 44]
46 Ops.AFTER dtypes.f32 [1, 26, 45]
47 Ops.CONST dtypes.weakint [] 128
48 Ops.MUL dtypes.weakint [26, 47]
49 Ops.ADD dtypes.weakint [48, 45]
50 Ops.INDEX dtypes.f32 [46, 49]
51 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
52 Ops.ADD dtypes.weakint [8, 44]
53 Ops.MUL dtypes.weakint [52, 2]
54 Ops.ADD dtypes.weakint [53, 16]
55 Ops.MUL dtypes.weakint [54, 2]
56 Ops.ADD dtypes.weakint [55, 21]
57 Ops.MUL dtypes.weakint [56, 2]
58 Ops.ADD dtypes.weakint [57, 25]
59 Ops.MUL dtypes.weakint [58, 47]
60 Ops.RANGE dtypes.weakint [19] (AxisType.WEAK, 0)
61 Ops.MUL dtypes.weakint [60, 2]
62 Ops.ADD dtypes.weakint [61, 9]
63 Ops.MUL dtypes.weakint [62, 2]
64 Ops.ADD dtypes.weakint [63, 37]
65 Ops.MUL dtypes.weakint [64, 2]
66 Ops.ADD dtypes.weakint [65, 41]
67 Ops.MUL dtypes.weakint [66, 2]
68 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 8)
69 Ops.ADD dtypes.weakint [67, 68]
70 Ops.ADD dtypes.weakint [59, 69]
71 Ops.INDEX dtypes.f16 [51, 70]
72 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
73 Ops.ADD dtypes.weakint [67, 44]
74 Ops.MUL dtypes.weakint [73, 47]
75 Ops.ADD dtypes.weakint [35, 16]
76 Ops.MUL dtypes.weakint [75, 2]
77 Ops.ADD dtypes.weakint [76, 21]
78 Ops.MUL dtypes.weakint [77, 2]
79 Ops.ADD dtypes.weakint [78, 25]
80 Ops.ADD dtypes.weakint [74, 79]
81 Ops.INDEX dtypes.f16 [72, 80]
82 Ops.CONST dtypes.weakfloat [] 0.0
83 Ops.CAST dtypes.f32 [82] dtypes.f32
84 Ops.STACK dtypes.f32 [83, 83, 83, 83]
85 Ops.WMMA dtypes.f32 [71, 81, 84] ((8, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
86 Ops.REDUCE dtypes.f32 [85, 60] (Ops.ADD, 0)
87 Ops.STORE dtypes.void [50, 86]
88 Ops.END dtypes.void [87, 3, 6, 9, 13, 28, 30, 33, 44]
89 Ops.SINK dtypes.void [88] KernelInfo(name="r_2_4_32_2_2_2_4_4_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST))), optimized=True)

