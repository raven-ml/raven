# tolk D212 D495 golden=1358c509f9225550ae340f811dcb87ed
0 Ops.CONST dtypes.weakint [] 1572864
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 10)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.CONST dtypes.weakint [] 2
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.RANGE dtypes.weakint [7] (AxisType.UPCAST, 7)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 5)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 768
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 6
18 Ops.RANGE dtypes.weakint [17] (AxisType.GLOBAL, 3)
19 Ops.CONST dtypes.weakint [] 16
20 Ops.MUL dtypes.weakint [18, 19]
21 Ops.RANGE dtypes.weakint [19] (AxisType.LOCAL, 9)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.MUL dtypes.weakint [22, 7]
24 Ops.RANGE dtypes.weakint [7] (AxisType.UPCAST, 6)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [25, 11]
27 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 4)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.ADD dtypes.weakint [16, 28]
30 Ops.RANGE dtypes.weakint [11] (AxisType.GLOBAL, 1)
31 Ops.CONST dtypes.weakint [] 393216
32 Ops.MUL dtypes.weakint [30, 31]
33 Ops.ADD dtypes.weakint [29, 32]
34 Ops.INDEX dtypes.f32 [1, 33]
35 Ops.CONST dtypes.weakint [] 32768
36 Ops.PARAM dtypes.f32 [35] ParamArg(slot=1, dtype=dtypes.f32)
37 Ops.MUL dtypes.weakint [14, 19]
38 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
39 Ops.MUL dtypes.weakint [38, 11]
40 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 8)
41 Ops.ADD dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [37, 41]
43 Ops.CONST dtypes.weakint [] 8192
44 Ops.MUL dtypes.weakint [30, 43]
45 Ops.ADD dtypes.weakint [42, 44]
46 Ops.INDEX dtypes.f32 [36, 45]
47 Ops.CONST dtypes.weakint [] 49152
48 Ops.PARAM dtypes.f32 [47] ParamArg(slot=2, dtype=dtypes.f32)
49 Ops.MUL dtypes.weakint [41, 15]
50 Ops.ADD dtypes.weakint [49, 28]
51 Ops.CONST dtypes.weakint [] 12288
52 Ops.MUL dtypes.weakint [30, 51]
53 Ops.ADD dtypes.weakint [50, 52]
54 Ops.INDEX dtypes.f32 [48, 53]
55 Ops.MUL dtypes.f32 [46, 54]
56 Ops.REDUCE dtypes.f32 [55, 38, 40] (Ops.ADD, 0)
57 Ops.PARAM dtypes.f32 [31] ParamArg(slot=3, dtype=dtypes.f32)
58 Ops.INDEX dtypes.f32 [57, 29]
59 Ops.ADD dtypes.f32 [56, 58]
60 Ops.STORE dtypes.void [34, 59]
61 Ops.END dtypes.void [60, 30, 3, 5, 9, 13, 18, 21, 24, 27]
62 Ops.SINK dtypes.void [61] KernelInfo(name="r_4_8_6_16_8_4_4_2_2_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=7, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL))), optimized=True)

