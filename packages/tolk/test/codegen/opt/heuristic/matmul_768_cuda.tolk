# tolk D212 golden=783b2cda92efc0e9605640c464fa31d3
0 Ops.CONST dtypes.weakint [] 1572864
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 9)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 6)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 4)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.CONST dtypes.weakint [] 768
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.CONST dtypes.weakint [] 6
19 Ops.RANGE dtypes.weakint [18] (AxisType.GLOBAL, 2)
20 Ops.CONST dtypes.weakint [] 16
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.RANGE dtypes.weakint [20] (AxisType.LOCAL, 8)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.MUL dtypes.weakint [23, 8]
25 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 5)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [26, 12]
28 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [17, 29]
31 Ops.INDEX dtypes.f32 [1, 30]
32 Ops.CONST dtypes.weakint [] 32768
33 Ops.PARAM dtypes.f32 [32] ParamArg(slot=1, dtype=dtypes.f32)
34 Ops.MUL dtypes.weakint [15, 20]
35 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
36 Ops.MUL dtypes.weakint [35, 12]
37 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 7)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.ADD dtypes.weakint [34, 38]
40 Ops.INDEX dtypes.f32 [33, 39]
41 Ops.CONST dtypes.weakint [] 12288
42 Ops.PARAM dtypes.f32 [41] ParamArg(slot=2, dtype=dtypes.f32)
43 Ops.MUL dtypes.weakint [38, 16]
44 Ops.ADD dtypes.weakint [43, 29]
45 Ops.INDEX dtypes.f32 [42, 44]
46 Ops.MUL dtypes.f32 [40, 45]
47 Ops.REDUCE dtypes.f32 [46, 35, 37] (Ops.ADD, 0)
48 Ops.STORE dtypes.void [31, 47]
49 Ops.END dtypes.void [48, 3, 6, 10, 14, 19, 22, 25, 28]
50 Ops.SINK dtypes.void [49] KernelInfo(name="r_32_6_16_8_4_4_2_2_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), optimized=True)

