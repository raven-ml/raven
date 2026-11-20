# tolk D73
0 Ops.CONST dtypes.weakint [] 23040
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1440
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 8
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.AFTER dtypes.f32 [1, 11]
13 Ops.INDEX dtypes.f32 [12, 11]
14 Ops.CONST dtypes.weakint [] 2880
15 Ops.PARAM dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32)
16 Ops.CONST dtypes.weakint [] 360
17 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 0, 0)
18 Ops.CONST dtypes.weakint [] 4
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.UPCAST, 5)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.CONST dtypes.weakint [] 1
24 Ops.RANGE dtypes.weakint [23] (AxisType.WEAK, 0, 1)
25 Ops.MUL dtypes.weakint [24, 4]
26 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [22, 27]
29 Ops.INDEX dtypes.f32 [15, 28]
30 Ops.CONST dtypes.weakint [] 512
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=2, dtype=dtypes.f32)
32 Ops.CONST dtypes.weakint [] 33177600
33 Ops.PARAM dtypes.u8 [32] ParamArg(slot=3, dtype=dtypes.u8)
34 Ops.MUL dtypes.weakint [11, 2]
35 Ops.ADD dtypes.weakint [34, 21]
36 Ops.INDEX dtypes.u8 [33, 35]
37 Ops.CAST dtypes.weakint [36] dtypes.weakint
38 Ops.MUL dtypes.weakint [37, 4]
39 Ops.ADD dtypes.weakint [38, 27]
40 Ops.INDEX dtypes.f32 [31, 39]
41 Ops.CONST dtypes.weakint [] 2073600
42 Ops.PARAM dtypes.f32 [41] ParamArg(slot=4, dtype=dtypes.f32)
43 Ops.CONST dtypes.weakint [] 90
44 Ops.MUL dtypes.weakint [11, 43]
45 Ops.CONST dtypes.weakint [] 16
46 Ops.DIV dtypes.weakint [21, 45]
47 Ops.ADD dtypes.weakint [44, 46]
48 Ops.INDEX dtypes.f32 [42, 47]
49 Ops.MUL dtypes.f32 [40, 48]
50 Ops.MUL dtypes.f32 [29, 49]
51 Ops.REDUCE dtypes.f32 [50, 17, 20, 24, 26] (Ops.ADD, 0)
52 Ops.STORE dtypes.void [13, 51]
53 Ops.END dtypes.void [52, 3, 6, 10]
54 Ops.SINK dtypes.void [53] KernelInfo(name="r_1440_8_2_360_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(4, AxisType.UPCAST))), optimized=True)

