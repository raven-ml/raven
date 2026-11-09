# tolk D357 D467 golden=f096858a3e3571b9affb03650cbe2d79
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 16
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 99
10 Ops.RANGE dtypes.weakint [9] (AxisType.WEAK, 0)
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.CONST dtypes.weakint [] 1571
15 Ops.CMPLT dtypes.bool [13, 14]
16 Ops.CONST dtypes.weakint [] 100544
17 Ops.PARAM dtypes.f32 [16] ParamArg(slot=1, dtype=dtypes.f32)
18 Ops.CONST dtypes.bool [] True
19 Ops.AND dtypes.bool [15, 18]
20 Ops.MUL dtypes.weakint [13, 0]
21 Ops.ADD dtypes.weakint [20, 7]
22 Ops.CONST dtypes.bool [] Invalid
23 Ops.WHERE dtypes.weakint [19, 21, 22]
24 Ops.INDEX dtypes.f32 [17, 23]
25 Ops.CONST dtypes.weakfloat [] 0.0
26 Ops.CAST dtypes.f32 [25] dtypes.f32
27 Ops.WHERE dtypes.f32 [15, 24, 26]
28 Ops.REDUCE dtypes.f32 [27, 10, 12] (Ops.ADD, 0)
29 Ops.STORE dtypes.void [8, 28]
30 Ops.END dtypes.void [29, 3, 6]
31 Ops.SINK dtypes.void [30] KernelInfo(name="r_4_16_16_99", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.PADTO, axis=2, arg=16), Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.LOCAL))), optimized=True)

