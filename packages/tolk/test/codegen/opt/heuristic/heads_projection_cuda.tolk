# tolk D212 golden=456a64b54adcc55760f045f24fe95fb3
0 Ops.CONST dtypes.weakint [] 1802240
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 4)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 11)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 55
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [9, 10]
12 Ops.RANGE dtypes.weakint [4] (AxisType.GLOBAL, 3)
13 Ops.CONST dtypes.weakint [] 4
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [13] (AxisType.LOCAL, 10)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 2
18 Ops.MUL dtypes.weakint [16, 17]
19 Ops.RANGE dtypes.weakint [17] (AxisType.UPCAST, 7)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.MUL dtypes.weakint [20, 13]
22 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 5)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.CONST dtypes.weakint [] 110
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [7, 17]
27 Ops.RANGE dtypes.weakint [17] (AxisType.UPCAST, 8)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.ADD dtypes.weakint [25, 28]
30 Ops.CONST dtypes.weakint [] 1
31 Ops.RANGE dtypes.weakint [30] (AxisType.GLOBAL, 2)
32 Ops.MUL dtypes.weakint [31, 13]
33 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 6)
34 Ops.ADD dtypes.weakint [32, 33]
35 Ops.CONST dtypes.weakint [] 28160
36 Ops.MUL dtypes.weakint [34, 35]
37 Ops.ADD dtypes.weakint [29, 36]
38 Ops.CONST dtypes.weakint [] 16
39 Ops.RANGE dtypes.weakint [38] (AxisType.GLOBAL, 1)
40 Ops.CONST dtypes.weakint [] 112640
41 Ops.MUL dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [37, 41]
43 Ops.CONST dtypes.bool [] Invalid
44 Ops.WHERE dtypes.weakint [11, 42, 43]
45 Ops.INDEX dtypes.f32 [1, 44]
46 Ops.CONST dtypes.weakint [] 262144
47 Ops.PARAM dtypes.f32 [46] ParamArg(slot=1, dtype=dtypes.f32)
48 Ops.CONST dtypes.weakint [] 64
49 Ops.MUL dtypes.weakint [23, 48]
50 Ops.RANGE dtypes.weakint [38] (AxisType.WEAK, 0)
51 Ops.MUL dtypes.weakint [50, 13]
52 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 9)
53 Ops.ADD dtypes.weakint [51, 52]
54 Ops.ADD dtypes.weakint [49, 53]
55 Ops.CONST dtypes.weakint [] 16384
56 Ops.MUL dtypes.weakint [39, 55]
57 Ops.ADD dtypes.weakint [54, 56]
58 Ops.INDEX dtypes.f32 [47, 57]
59 Ops.PARAM dtypes.f32 [35] ParamArg(slot=2, dtype=dtypes.f32)
60 Ops.MUL dtypes.weakint [34, 24]
61 Ops.ADD dtypes.weakint [60, 28]
62 Ops.CONST dtypes.weakint [] 440
63 Ops.MUL dtypes.weakint [53, 62]
64 Ops.ADD dtypes.weakint [61, 63]
65 Ops.WHERE dtypes.weakint [11, 64, 43]
66 Ops.INDEX dtypes.f32 [59, 65]
67 Ops.MUL dtypes.f32 [58, 66]
68 Ops.REDUCE dtypes.f32 [67, 50, 52] (Ops.ADD, 0)
69 Ops.STORE dtypes.void [45, 68]
70 Ops.END dtypes.void [69, 39, 31, 33, 12, 15, 19, 22, 3, 6, 27]
71 Ops.SINK dtypes.void [70] KernelInfo(name="r_16_8_7_4_8_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=7, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=2, arg=8), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.LOCAL))), optimized=True)

