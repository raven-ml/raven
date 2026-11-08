# tolk D280 golden=2069c6f6f4ac3164bae5d9cc038bc266
0 Ops.CONST dtypes.weakint [] 7
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.CONST dtypes.weakint [] 6
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 0, 0)
7 Ops.CONST dtypes.weakint [] 4
8 Ops.CMPLT dtypes.bool [6, 7]
9 Ops.CONST dtypes.weakint [] 35
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 5
12 Ops.MUL dtypes.weakint [2, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0, 1)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.INDEX dtypes.f32 [10, 14]
16 Ops.CONST dtypes.weakfloat [] 1.0
17 Ops.CAST dtypes.f32 [16] dtypes.f32
18 Ops.WHERE dtypes.f32 [8, 15, 17]
19 Ops.REDUCE dtypes.f32 [18, 6, 13] (Ops.MUL, 0)
20 Ops.STORE dtypes.void [4, 19]
21 Ops.END dtypes.void [20, 2]
22 Ops.SINK dtypes.void [21] KernelInfo(name="r_7_6_5", applied_opts=(Opt(op=OptOps.PADTO, axis=1, arg=3),), optimized=True)

