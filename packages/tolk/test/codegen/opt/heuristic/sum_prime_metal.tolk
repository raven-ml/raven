# tolk D467 golden=e2b3af7bae1bfc594c879ec31c9e5101
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.GLOBAL, 1)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 99
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
6 Ops.CONST dtypes.weakint [] 16
7 Ops.MUL dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [6] (AxisType.LOCAL, 2)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 1571
11 Ops.CMPLT dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 100544
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.CONST dtypes.bool [] True
15 Ops.AND dtypes.bool [11, 14]
16 Ops.MUL dtypes.weakint [2, 10]
17 Ops.ADD dtypes.weakint [16, 9]
18 Ops.CONST dtypes.bool [] Invalid
19 Ops.WHERE dtypes.weakint [15, 17, 18]
20 Ops.INDEX dtypes.f32 [13, 19]
21 Ops.CONST dtypes.weakfloat [] 0.0
22 Ops.CAST dtypes.f32 [21] dtypes.f32
23 Ops.WHERE dtypes.f32 [11, 20, 22]
24 Ops.REDUCE dtypes.f32 [23, 5, 8] (Ops.ADD, 0)
25 Ops.STORE dtypes.void [3, 24]
26 Ops.END dtypes.void [25, 2]
27 Ops.SINK dtypes.void [26] KernelInfo(name="r_64_16_99", applied_opts=(Opt(op=OptOps.PADTO, axis=1, arg=16), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL))), optimized=True)

