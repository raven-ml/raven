# tolk D280 golden=1d6fb5480cf26e06dc68621a279ebe76
0 Ops.CONST dtypes.weakint [] 2048
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 128
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 4)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 2)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.AFTER dtypes.f32 [1, 16]
18 Ops.INDEX dtypes.f32 [17, 16]
19 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.INDEX dtypes.f32 [19, 16]
21 Ops.SQRT dtypes.f32 [20]
22 Ops.CONST dtypes.weakint [] 262144
23 Ops.PARAM dtypes.f32 [22] ParamArg(slot=2, dtype=dtypes.f32)
24 Ops.MUL dtypes.weakint [16, 2]
25 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.INDEX dtypes.f32 [23, 26]
28 Ops.CONST dtypes.weakfloat [] 1.0
29 Ops.ADD dtypes.f32 [27, 28]
30 Ops.REDUCE dtypes.f32 [29, 25] (Ops.ADD, 0)
31 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
32 Ops.MUL dtypes.f32 [30, 31]
33 Ops.EXP2 dtypes.f32 [32]
34 Ops.ADD dtypes.f32 [21, 33]
35 Ops.STORE dtypes.void [18, 34]
36 Ops.END dtypes.void [35, 3, 6, 9, 12, 15]
37 Ops.SINK dtypes.void [36] KernelInfo(name="r_128_2_2_2_2_128", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL))), optimized=True)

