# tolk D89 golden=062629688ba5f108d9adb423ae3fef8a
0 Ops.CONST dtypes.weakint [] 768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 3
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 256
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 16
7 Ops.RANGE dtypes.weakint [6] (AxisType.GLOBAL, 3)
8 Ops.MUL dtypes.weakint [7, 6]
9 Ops.RANGE dtypes.weakint [6] (AxisType.LOCAL, 4)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.ADD dtypes.weakint [5, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 196608
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 64
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0)
17 Ops.MUL dtypes.weakint [16, 6]
18 Ops.RANGE dtypes.weakint [6] (AxisType.LOCAL, 5)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.MUL dtypes.weakint [19, 2]
21 Ops.ADD dtypes.weakint [20, 3]
22 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 1)
23 Ops.CONST dtypes.weakint [] 3072
24 Ops.MUL dtypes.weakint [22, 23]
25 Ops.ADD dtypes.weakint [21, 24]
26 Ops.INDEX dtypes.f32 [14, 25]
27 Ops.CONST dtypes.weakint [] 16777216
28 Ops.PARAM dtypes.f32 [27] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [22, 4]
30 Ops.ADD dtypes.weakint [29, 10]
31 Ops.CONST dtypes.weakint [] 16384
32 Ops.MUL dtypes.weakint [19, 31]
33 Ops.ADD dtypes.weakint [30, 32]
34 Ops.INDEX dtypes.f32 [28, 33]
35 Ops.ADD dtypes.f32 [26, 34]
36 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
37 Ops.MUL dtypes.f32 [35, 36]
38 Ops.EXP2 dtypes.f32 [37]
39 Ops.REDUCE dtypes.f32 [38, 16, 18, 22] (Ops.ADD, 0)
40 Ops.STORE dtypes.void [12, 39]
41 Ops.END dtypes.void [40, 3, 7, 9]
42 Ops.SINK dtypes.void [41] KernelInfo(name="r_3_16_16_16_64_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=3, arg=(16, AxisType.LOCAL))), optimized=True)

