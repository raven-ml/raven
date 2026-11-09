# tolk D357 D467 golden=03f163c1ed6f2fed85d4cd3a979f641e
0 Ops.CONST dtypes.weakint [] 8
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.GLOBAL, 2)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 17
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
6 Ops.CONST dtypes.weakint [] 16
7 Ops.MUL dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [6] (AxisType.LOCAL, 3)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 257
11 Ops.CMPLT dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 131584
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.CONST dtypes.bool [] True
15 Ops.AND dtypes.bool [11, 14]
16 Ops.MUL dtypes.weakint [2, 10]
17 Ops.ADD dtypes.weakint [16, 9]
18 Ops.CONST dtypes.weakint [] 64
19 Ops.RANGE dtypes.weakint [18] (AxisType.WEAK, 0)
20 Ops.CONST dtypes.weakint [] 2056
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.ADD dtypes.weakint [17, 21]
23 Ops.CONST dtypes.bool [] Invalid
24 Ops.WHERE dtypes.weakint [15, 22, 23]
25 Ops.INDEX dtypes.f32 [13, 24]
26 Ops.CONST dtypes.weakfloat [] 0.0
27 Ops.CAST dtypes.f32 [26] dtypes.f32
28 Ops.WHERE dtypes.f32 [11, 25, 27]
29 Ops.REDUCE dtypes.f32 [28, 19, 5, 8] (Ops.ADD, 0)
30 Ops.STORE dtypes.void [3, 29]
31 Ops.END dtypes.void [30, 2]
32 Ops.SINK dtypes.void [31] KernelInfo(name="r_8_16_64_17", applied_opts=(Opt(op=OptOps.PADTO, axis=2, arg=16), Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.LOCAL))), optimized=True)

