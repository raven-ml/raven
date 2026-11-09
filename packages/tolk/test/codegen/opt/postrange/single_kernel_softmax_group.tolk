# tolk D101 golden=c19d01d085681016def940b02cea44b3
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 3)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 4)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.INDEX dtypes.f32 [1, 6]
8 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
9 Ops.INDEX dtypes.f32 [8, 6]
10 Ops.CONST dtypes.weakint [] 8
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 0)
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.LOCAL, 5)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [4, 15]
17 Ops.INDEX dtypes.f32 [8, 16]
18 Ops.REDUCE dtypes.f32 [17, 11, 14] (Ops.MAX, 0)
19 Ops.CONST dtypes.weakfloat [] -1.0
20 Ops.MUL dtypes.f32 [18, 19]
21 Ops.ADD dtypes.f32 [9, 20]
22 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
23 Ops.MUL dtypes.f32 [21, 22]
24 Ops.EXP2 dtypes.f32 [23]
25 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
26 Ops.ADD dtypes.weakint [4, 25]
27 Ops.INDEX dtypes.f32 [8, 26]
28 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 1)
29 Ops.MUL dtypes.weakint [28, 12]
30 Ops.ADD dtypes.weakint [29, 14]
31 Ops.ADD dtypes.weakint [4, 30]
32 Ops.INDEX dtypes.f32 [8, 31]
33 Ops.REDUCE dtypes.f32 [32, 28, 14] (Ops.MAX, 0)
34 Ops.MUL dtypes.f32 [33, 19]
35 Ops.ADD dtypes.f32 [27, 34]
36 Ops.MUL dtypes.f32 [35, 22]
37 Ops.EXP2 dtypes.f32 [36]
38 Ops.REDUCE dtypes.f32 [37, 25] (Ops.ADD, 0)
39 Ops.RECIP dtypes.f32 [38]
40 Ops.MUL dtypes.f32 [24, 39]
41 Ops.STORE dtypes.void [7, 40]
42 Ops.END dtypes.void [41, 3, 5]
43 Ops.SINK dtypes.void [42] KernelInfo(name="r_32_32_4_8_8_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.LOCAL)),), optimized=True)

