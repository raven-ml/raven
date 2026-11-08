# tolk D280 golden=44988413a23585abf80718ba97dafb13
0 Ops.CONST dtypes.weakint [] 3584
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 7
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
9 Ops.CONST dtypes.weakint [] 8
10 Ops.MUL dtypes.weakint [8, 9]
11 Ops.RANGE dtypes.weakint [9] (AxisType.LOCAL, 5)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.CONST dtypes.weakint [] 4
14 Ops.RANGE dtypes.weakint [13] (AxisType.GLOBAL, 0)
15 Ops.CONST dtypes.weakint [] 16
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.RANGE dtypes.weakint [15] (AxisType.LOCAL, 4)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.AFTER dtypes.f32 [1, 7, 12, 18]
20 Ops.MUL dtypes.weakint [7, 9]
21 Ops.ADD dtypes.weakint [20, 12]
22 Ops.CONST dtypes.weakint [] 56
23 Ops.MUL dtypes.weakint [18, 22]
24 Ops.ADD dtypes.weakint [21, 23]
25 Ops.INDEX dtypes.f32 [19, 24]
26 Ops.CMPLT dtypes.bool [7, 2]
27 Ops.CONST dtypes.bool [] False
28 Ops.CMPEQ dtypes.bool [26, 27]
29 Ops.CONST dtypes.weakint [] 6
30 Ops.CMPLT dtypes.bool [7, 29]
31 Ops.AND dtypes.bool [28, 30]
32 Ops.CMPLT dtypes.bool [12, 2]
33 Ops.CMPEQ dtypes.bool [32, 27]
34 Ops.CMPLT dtypes.bool [12, 4]
35 Ops.AND dtypes.bool [33, 34]
36 Ops.AND dtypes.bool [31, 35]
37 Ops.CONST dtypes.weakint [] 1920
38 Ops.PARAM dtypes.f32 [37] ParamArg(slot=1, dtype=dtypes.f32)
39 Ops.AND dtypes.bool [35, 31]
40 Ops.MUL dtypes.weakint [7, 29]
41 Ops.ADD dtypes.weakint [40, 12]
42 Ops.CONST dtypes.weakint [] 30
43 Ops.MUL dtypes.weakint [18, 42]
44 Ops.ADD dtypes.weakint [41, 43]
45 Ops.CONST dtypes.weakint [] -7
46 Ops.ADD dtypes.weakint [44, 45]
47 Ops.CONST dtypes.bool [] Invalid
48 Ops.WHERE dtypes.weakint [39, 46, 47]
49 Ops.INDEX dtypes.f32 [38, 48]
50 Ops.CONST dtypes.weakfloat [] 0.0
51 Ops.WHERE dtypes.f32 [36, 49, 50]
52 Ops.CONST dtypes.weakfloat [] 1.0
53 Ops.ADD dtypes.f32 [51, 52]
54 Ops.STORE dtypes.void [25, 53]
55 Ops.END dtypes.void [54, 14, 17, 3, 6, 8, 11]
56 Ops.SINK dtypes.void [55] KernelInfo(name="E_4_16_8_7", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL))), optimized=True)

