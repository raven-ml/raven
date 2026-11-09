# tolk D89 golden=281acca670c72d8a4b2d7798d51ac7ad
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 256
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 1
12 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0, 0)
13 Ops.CONST dtypes.weakint [] 8
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 4)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 32
18 Ops.MUL dtypes.weakint [16, 17]
19 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0, 1)
20 Ops.MUL dtypes.weakint [19, 17]
21 Ops.RANGE dtypes.weakint [17] (AxisType.LOCAL, 2)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.ADD dtypes.weakint [18, 22]
24 Ops.INDEX dtypes.f32 [10, 23]
25 Ops.CONST dtypes.weakint [] 17408
26 Ops.PARAM dtypes.u8 [25] ParamArg(slot=2, dtype=dtypes.u8)
27 Ops.CONST dtypes.weakint [] 34
28 Ops.MUL dtypes.weakint [16, 27]
29 Ops.ADD dtypes.weakint [28, 22]
30 Ops.CONST dtypes.weakint [] 272
31 Ops.MUL dtypes.weakint [7, 30]
32 Ops.ADD dtypes.weakint [29, 31]
33 Ops.CONST dtypes.weakint [] 2
34 Ops.ADD dtypes.weakint [32, 33]
35 Ops.INDEX dtypes.u8 [26, 34]
36 Ops.BITCAST dtypes.i8 [35] dtypes.i8
37 Ops.CAST dtypes.f32 [36] dtypes.f32
38 Ops.ADD dtypes.weakint [28, 31]
39 Ops.INDEX dtypes.u8 [26, 38]
40 Ops.CAST dtypes.u16 [39] dtypes.u16
41 Ops.ADD dtypes.weakint [38, 11]
42 Ops.INDEX dtypes.u8 [26, 41]
43 Ops.CAST dtypes.u16 [42] dtypes.u16
44 Ops.MUL dtypes.u16 [43, 9]
45 Ops.OR dtypes.u16 [40, 44]
46 Ops.BITCAST dtypes.f16 [45] dtypes.f16
47 Ops.CAST dtypes.f32 [46] dtypes.f32
48 Ops.MUL dtypes.f32 [37, 47]
49 Ops.MUL dtypes.f32 [24, 48]
50 Ops.REDUCE dtypes.f32 [49, 12, 15, 19, 21] (Ops.ADD, 0)
51 Ops.STORE dtypes.void [8, 50]
52 Ops.END dtypes.void [51, 3, 6]
53 Ops.SINK dtypes.void [52] KernelInfo(name="r_16_32_4_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=3, arg=(8, AxisType.UPCAST))), optimized=True)

