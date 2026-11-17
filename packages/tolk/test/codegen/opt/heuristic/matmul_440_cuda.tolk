# tolk D212 golden=3c8b06796c57773e2e03e003032aaa02
0 Ops.CONST dtypes.weakint [] 112640
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 8)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 55
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 2
13 Ops.RANGE dtypes.weakint [12] (AxisType.GLOBAL, 1)
14 Ops.CONST dtypes.weakint [] 16
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.LOCAL, 9)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.MUL dtypes.weakint [17, 12]
19 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 6)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.CONST dtypes.weakint [] 4
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 4)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.CONST dtypes.weakint [] 440
26 Ops.MUL dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [7, 12]
28 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 5)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.MUL dtypes.weakint [29, 21]
31 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 3)
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.ADD dtypes.weakint [26, 32]
34 Ops.CONST dtypes.bool [] Invalid
35 Ops.WHERE dtypes.weakint [11, 33, 34]
36 Ops.INDEX dtypes.f32 [1, 35]
37 Ops.CONST dtypes.weakint [] 16384
38 Ops.PARAM dtypes.f32 [37] ParamArg(slot=1, dtype=dtypes.f32)
39 Ops.CONST dtypes.weakint [] 64
40 Ops.MUL dtypes.weakint [24, 39]
41 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 0)
42 Ops.MUL dtypes.weakint [41, 21]
43 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 7)
44 Ops.ADD dtypes.weakint [42, 43]
45 Ops.ADD dtypes.weakint [40, 44]
46 Ops.INDEX dtypes.f32 [38, 45]
47 Ops.CONST dtypes.weakint [] 28160
48 Ops.PARAM dtypes.f32 [47] ParamArg(slot=2, dtype=dtypes.f32)
49 Ops.MUL dtypes.weakint [44, 25]
50 Ops.ADD dtypes.weakint [49, 32]
51 Ops.WHERE dtypes.weakint [11, 50, 34]
52 Ops.INDEX dtypes.f32 [48, 51]
53 Ops.MUL dtypes.f32 [46, 52]
54 Ops.REDUCE dtypes.f32 [53, 41, 43] (Ops.ADD, 0)
55 Ops.STORE dtypes.void [36, 54]
56 Ops.END dtypes.void [55, 13, 16, 19, 23, 3, 6, 28, 31]
57 Ops.SINK dtypes.void [56] KernelInfo(name="r_2_7_8_16_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=1, arg=8), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL))), optimized=True)

