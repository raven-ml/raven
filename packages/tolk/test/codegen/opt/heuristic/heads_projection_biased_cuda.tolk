# tolk D212 golden=7b3e4d0987aa43f9e89c3e99f69dabe4
0 Ops.CONST dtypes.weakint [] 1802240
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 4)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 10)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 55
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 2
13 Ops.RANGE dtypes.weakint [12] (AxisType.GLOBAL, 3)
14 Ops.CONST dtypes.weakint [] 16
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.LOCAL, 11)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.MUL dtypes.weakint [17, 12]
19 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 7)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.CONST dtypes.weakint [] 4
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.CONST dtypes.weakint [] 110
26 Ops.MUL dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [7, 12]
28 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 8)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [26, 29]
31 Ops.CONST dtypes.weakint [] 1
32 Ops.RANGE dtypes.weakint [31] (AxisType.GLOBAL, 2)
33 Ops.MUL dtypes.weakint [32, 21]
34 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 6)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.CONST dtypes.weakint [] 28160
37 Ops.MUL dtypes.weakint [35, 36]
38 Ops.ADD dtypes.weakint [30, 37]
39 Ops.RANGE dtypes.weakint [14] (AxisType.GLOBAL, 1)
40 Ops.CONST dtypes.weakint [] 112640
41 Ops.MUL dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [38, 41]
43 Ops.CONST dtypes.bool [] Invalid
44 Ops.WHERE dtypes.weakint [11, 42, 43]
45 Ops.INDEX dtypes.f32 [1, 44]
46 Ops.CONST dtypes.weakint [] 262144
47 Ops.PARAM dtypes.f32 [46] ParamArg(slot=1, dtype=dtypes.f32)
48 Ops.CONST dtypes.weakint [] 64
49 Ops.MUL dtypes.weakint [24, 48]
50 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 0)
51 Ops.MUL dtypes.weakint [50, 21]
52 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 9)
53 Ops.ADD dtypes.weakint [51, 52]
54 Ops.ADD dtypes.weakint [49, 53]
55 Ops.CONST dtypes.weakint [] 16384
56 Ops.MUL dtypes.weakint [39, 55]
57 Ops.ADD dtypes.weakint [54, 56]
58 Ops.INDEX dtypes.f32 [47, 57]
59 Ops.PARAM dtypes.f32 [36] ParamArg(slot=2, dtype=dtypes.f32)
60 Ops.MUL dtypes.weakint [35, 25]
61 Ops.ADD dtypes.weakint [60, 29]
62 Ops.CONST dtypes.weakint [] 440
63 Ops.MUL dtypes.weakint [53, 62]
64 Ops.ADD dtypes.weakint [61, 63]
65 Ops.WHERE dtypes.weakint [11, 64, 43]
66 Ops.INDEX dtypes.f32 [59, 65]
67 Ops.MUL dtypes.f32 [58, 66]
68 Ops.REDUCE dtypes.f32 [67, 50, 52] (Ops.ADD, 0)
69 Ops.PARAM dtypes.f32 [21] ParamArg(slot=3, dtype=dtypes.f32)
70 Ops.INDEX dtypes.f32 [69, 35]
71 Ops.ADD dtypes.f32 [68, 70]
72 Ops.STORE dtypes.void [45, 71]
73 Ops.END dtypes.void [72, 39, 32, 34, 13, 16, 19, 23, 3, 6, 28]
74 Ops.SINK dtypes.void [73] KernelInfo(name="r_16_2_7_8_16_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=7, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=2, arg=8), Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL))), optimized=True)

