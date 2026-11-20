# tolk D73
0 Ops.CONST dtypes.weakint [] 23040
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2880
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.AFTER dtypes.f32 [1, 11]
13 Ops.INDEX dtypes.f32 [12, 11]
14 Ops.PARAM dtypes.f32 [2] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 180
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0, 0)
17 Ops.CONST dtypes.weakint [] 8
18 Ops.MUL dtypes.weakint [16, 17]
19 Ops.RANGE dtypes.weakint [17] (AxisType.UPCAST, 5)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.MUL dtypes.weakint [20, 4]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0, 1)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.ADD dtypes.weakint [21, 26]
28 Ops.INDEX dtypes.f32 [14, 27]
29 Ops.CONST dtypes.weakint [] 512
30 Ops.PARAM dtypes.f32 [29] ParamArg(slot=2, dtype=dtypes.f32)
31 Ops.CONST dtypes.weakint [] 33177600
32 Ops.PARAM dtypes.u8 [31] ParamArg(slot=3, dtype=dtypes.u8)
33 Ops.CONST dtypes.weakint [] 1440
34 Ops.MUL dtypes.weakint [11, 33]
35 Ops.ADD dtypes.weakint [34, 20]
36 Ops.INDEX dtypes.u8 [32, 35]
37 Ops.CAST dtypes.weakint [36] dtypes.weakint
38 Ops.MUL dtypes.weakint [37, 4]
39 Ops.ADD dtypes.weakint [38, 26]
40 Ops.INDEX dtypes.f32 [30, 39]
41 Ops.CONST dtypes.weakint [] 2073600
42 Ops.PARAM dtypes.f32 [41] ParamArg(slot=4, dtype=dtypes.f32)
43 Ops.CONST dtypes.weakint [] 90
44 Ops.MUL dtypes.weakint [11, 43]
45 Ops.CONST dtypes.weakint [] 16
46 Ops.DIV dtypes.weakint [20, 45]
47 Ops.ADD dtypes.weakint [44, 46]
48 Ops.INDEX dtypes.f32 [42, 47]
49 Ops.MUL dtypes.f32 [40, 48]
50 Ops.MUL dtypes.f32 [28, 49]
51 Ops.REDUCE dtypes.f32 [50, 16, 19, 23, 25] (Ops.ADD, 0)
52 Ops.STORE dtypes.void [13, 51]
53 Ops.END dtypes.void [52, 3, 6, 10]
54 Ops.SINK dtypes.void [53] KernelInfo(name="r_2880_4_2_180_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(8, AxisType.UPCAST))), optimized=True)

