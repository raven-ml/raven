# tolk D503 golden=6b5ab781f83221fc229411dce494bf83
0 Ops.CONST dtypes.weakint [] 3136
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.CONST dtypes.weakint [] 1
6 Ops.RANGE dtypes.weakint [5] (AxisType.GLOBAL, 2)
7 Ops.MUL dtypes.weakint [6, 2]
8 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.ADD dtypes.weakint [4, 9]
11 Ops.CONST dtypes.weakint [] 2
12 Ops.RANGE dtypes.weakint [11] (AxisType.GLOBAL, 0)
13 Ops.CONST dtypes.weakint [] 32
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [13] (AxisType.LOCAL, 4)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 49
18 Ops.MUL dtypes.weakint [16, 17]
19 Ops.ADD dtypes.weakint [10, 18]
20 Ops.INDEX dtypes.f32 [1, 19]
21 Ops.CMPLT dtypes.bool [3, 5]
22 Ops.CONST dtypes.bool [] False
23 Ops.CMPEQ dtypes.bool [21, 22]
24 Ops.CONST dtypes.weakint [] 6
25 Ops.CMPLT dtypes.bool [3, 24]
26 Ops.AND dtypes.bool [23, 25]
27 Ops.CMPLT dtypes.bool [9, 5]
28 Ops.CMPEQ dtypes.bool [27, 22]
29 Ops.CMPLT dtypes.bool [9, 24]
30 Ops.AND dtypes.bool [28, 29]
31 Ops.AND dtypes.bool [26, 30]
32 Ops.CONST dtypes.weakint [] 1600
33 Ops.PARAM dtypes.f32 [32] ParamArg(slot=1, dtype=dtypes.f32)
34 Ops.AND dtypes.bool [30, 26]
35 Ops.CONST dtypes.weakint [] 5
36 Ops.MUL dtypes.weakint [3, 35]
37 Ops.ADD dtypes.weakint [36, 9]
38 Ops.CONST dtypes.weakint [] 25
39 Ops.MUL dtypes.weakint [16, 38]
40 Ops.ADD dtypes.weakint [37, 39]
41 Ops.CONST dtypes.weakint [] -6
42 Ops.ADD dtypes.weakint [40, 41]
43 Ops.CONST dtypes.bool [] Invalid
44 Ops.WHERE dtypes.weakint [34, 42, 43]
45 Ops.INDEX dtypes.f32 [33, 44]
46 Ops.CONST dtypes.weakfloat [] 0.0
47 Ops.WHERE dtypes.f32 [31, 45, 46]
48 Ops.CONST dtypes.weakfloat [] 1.0
49 Ops.ADD dtypes.f32 [47, 48]
50 Ops.STORE dtypes.void [20, 49]
51 Ops.END dtypes.void [50, 12, 15, 3, 6, 8]
52 Ops.SINK dtypes.void [51] KernelInfo(name="E_2_7_32_7", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL))), optimized=True)

