# tolk D188 golden=18f0eb19416db23d1227ea4ff5997508
0 Ops.CONST dtypes.weakint [] 131072
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 128
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 3)
12 Ops.CONST dtypes.weakint [] 8
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 4)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 1)
18 Ops.CONST dtypes.weakint [] 16384
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.ADD dtypes.weakint [16, 19]
21 Ops.INDEX dtypes.f32 [1, 20]
22 Ops.PARAM dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32)
23 Ops.MUL dtypes.weakint [7, 10]
24 Ops.CONST dtypes.weakint [] 1
25 Ops.RANGE dtypes.weakint [24] (AxisType.WEAK, 0)
26 Ops.MUL dtypes.weakint [25, 10]
27 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 6)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.ADD dtypes.weakint [23, 28]
30 Ops.CONST dtypes.weakint [] 2048
31 Ops.MUL dtypes.weakint [17, 30]
32 Ops.ADD dtypes.weakint [29, 31]
33 Ops.INDEX dtypes.f32 [22, 32]
34 Ops.PARAM dtypes.f32 [18] ParamArg(slot=2, dtype=dtypes.f32)
35 Ops.MUL dtypes.weakint [15, 10]
36 Ops.ADD dtypes.weakint [35, 28]
37 Ops.ADD dtypes.weakint [36, 31]
38 Ops.INDEX dtypes.f32 [34, 37]
39 Ops.MUL dtypes.f32 [33, 38]
40 Ops.REDUCE dtypes.f32 [39, 25, 27] (Ops.ADD, 0)
41 Ops.STORE dtypes.void [21, 40]
42 Ops.END dtypes.void [41, 17, 3, 6, 11, 14]
43 Ops.SINK dtypes.void [42] KernelInfo(name="r_8_32_16_8_4_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST))), optimized=True)

