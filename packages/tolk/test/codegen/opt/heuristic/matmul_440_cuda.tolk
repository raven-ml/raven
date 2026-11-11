# tolk D212 golden=3c8b06796c57773e2e03e003032aaa02
0 Ops.CONST dtypes.weakint [] 112640
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 9)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 55
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [9, 10]
12 Ops.RANGE dtypes.weakint [4] (AxisType.GLOBAL, 1)
13 Ops.CONST dtypes.weakint [] 4
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [13] (AxisType.LOCAL, 8)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 2
18 Ops.MUL dtypes.weakint [16, 17]
19 Ops.RANGE dtypes.weakint [17] (AxisType.UPCAST, 6)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.MUL dtypes.weakint [20, 13]
22 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 4)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.CONST dtypes.weakint [] 440
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [7, 17]
27 Ops.RANGE dtypes.weakint [17] (AxisType.UPCAST, 5)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.MUL dtypes.weakint [28, 13]
30 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 3)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.ADD dtypes.weakint [25, 31]
33 Ops.CONST dtypes.bool [] Invalid
34 Ops.WHERE dtypes.weakint [11, 32, 33]
35 Ops.INDEX dtypes.f32 [1, 34]
36 Ops.CONST dtypes.weakint [] 16384
37 Ops.PARAM dtypes.f32 [36] ParamArg(slot=1, dtype=dtypes.f32)
38 Ops.CONST dtypes.weakint [] 64
39 Ops.MUL dtypes.weakint [23, 38]
40 Ops.CONST dtypes.weakint [] 16
41 Ops.RANGE dtypes.weakint [40] (AxisType.WEAK, 0)
42 Ops.MUL dtypes.weakint [41, 13]
43 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 7)
44 Ops.ADD dtypes.weakint [42, 43]
45 Ops.ADD dtypes.weakint [39, 44]
46 Ops.INDEX dtypes.f32 [37, 45]
47 Ops.CONST dtypes.weakint [] 28160
48 Ops.PARAM dtypes.f32 [47] ParamArg(slot=2, dtype=dtypes.f32)
49 Ops.MUL dtypes.weakint [44, 24]
50 Ops.ADD dtypes.weakint [49, 31]
51 Ops.WHERE dtypes.weakint [11, 50, 33]
52 Ops.INDEX dtypes.f32 [48, 51]
53 Ops.MUL dtypes.f32 [46, 52]
54 Ops.REDUCE dtypes.f32 [53, 41, 43] (Ops.ADD, 0)
55 Ops.STORE dtypes.void [35, 54]
56 Ops.END dtypes.void [55, 12, 15, 19, 22, 3, 6, 27, 30]
57 Ops.SINK dtypes.void [56] KernelInfo(name="r_8_7_4_8_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=1, arg=8), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL))), optimized=True)

