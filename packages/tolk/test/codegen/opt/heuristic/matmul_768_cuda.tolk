# tolk D212 golden=783b2cda92efc0e9605640c464fa31d3
0 Ops.CONST dtypes.weakint [] 1572864
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 128
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 8)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 4)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 768
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 6
18 Ops.RANGE dtypes.weakint [17] (AxisType.GLOBAL, 2)
19 Ops.CONST dtypes.weakint [] 16
20 Ops.MUL dtypes.weakint [18, 19]
21 Ops.RANGE dtypes.weakint [19] (AxisType.LOCAL, 9)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.MUL dtypes.weakint [22, 4]
24 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [25, 11]
27 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 3)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.ADD dtypes.weakint [16, 28]
30 Ops.INDEX dtypes.f32 [1, 29]
31 Ops.CONST dtypes.weakint [] 32768
32 Ops.PARAM dtypes.f32 [31] ParamArg(slot=1, dtype=dtypes.f32)
33 Ops.MUL dtypes.weakint [14, 19]
34 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
35 Ops.MUL dtypes.weakint [34, 11]
36 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 7)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.ADD dtypes.weakint [33, 37]
39 Ops.INDEX dtypes.f32 [32, 38]
40 Ops.CONST dtypes.weakint [] 12288
41 Ops.PARAM dtypes.f32 [40] ParamArg(slot=2, dtype=dtypes.f32)
42 Ops.MUL dtypes.weakint [37, 15]
43 Ops.ADD dtypes.weakint [42, 28]
44 Ops.INDEX dtypes.f32 [41, 43]
45 Ops.MUL dtypes.f32 [39, 44]
46 Ops.REDUCE dtypes.f32 [45, 34, 36] (Ops.ADD, 0)
47 Ops.STORE dtypes.void [30, 46]
48 Ops.END dtypes.void [47, 3, 6, 9, 13, 18, 21, 24, 27]
49 Ops.SINK dtypes.void [48] KernelInfo(name="r_128_6_2_16_4_4_2_2_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL))), optimized=True)

