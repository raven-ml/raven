# tolk D280 golden=ba7231e79320b0d6e30555b8e22f02ff
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 4)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.CONST dtypes.weakint [] 64
15 Ops.RANGE dtypes.weakint [14] (AxisType.GLOBAL, 2)
16 Ops.MUL dtypes.weakint [15, 2]
17 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 8)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.AFTER dtypes.f32 [1, 13, 18]
20 Ops.CONST dtypes.weakint [] 128
21 Ops.MUL dtypes.weakint [13, 20]
22 Ops.ADD dtypes.weakint [21, 18]
23 Ops.INDEX dtypes.f32 [19, 22]
24 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
25 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 5)
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.CONST dtypes.weakint [] 16
30 Ops.MUL dtypes.weakint [28, 29]
31 Ops.RANGE dtypes.weakint [29] (AxisType.WEAK, 0)
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.ADD dtypes.weakint [21, 32]
34 Ops.INDEX dtypes.f32 [24, 33]
35 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
36 Ops.MUL dtypes.weakint [32, 20]
37 Ops.ADD dtypes.weakint [36, 18]
38 Ops.INDEX dtypes.f32 [35, 37]
39 Ops.MUL dtypes.f32 [34, 38]
40 Ops.REDUCE dtypes.f32 [39, 25, 27, 31] (Ops.ADD, 0)
41 Ops.STORE dtypes.void [23, 40]
42 Ops.END dtypes.void [41, 3, 6, 9, 12, 15, 17]
43 Ops.SINK dtypes.void [42] KernelInfo(name="r_2_64_4_4_2_4_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=4, arg=(8, AxisType.LOCAL, True)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST))), optimized=True)

