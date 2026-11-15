# tolk D490 golden=bc09c3983fea1f752cafd80285006e18
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.INDEX dtypes.f32 [1, 6]
8 Ops.CONST dtypes.weakint [] 8
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
10 Ops.CMPLT dtypes.bool [9, 2]
11 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.bool [] True
13 Ops.AND dtypes.bool [10, 12]
14 Ops.ADD dtypes.weakint [4, 9]
15 Ops.CONST dtypes.bool [] Invalid
16 Ops.WHERE dtypes.weakint [13, 14, 15]
17 Ops.INDEX dtypes.f32 [11, 16]
18 Ops.CONST dtypes.weakfloat [] 0.0
19 Ops.CAST dtypes.f32 [18] dtypes.f32
20 Ops.WHERE dtypes.f32 [10, 17, 19]
21 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
22 Ops.MUL dtypes.weakint [9, 2]
23 Ops.ADD dtypes.weakint [22, 5]
24 Ops.WHERE dtypes.weakint [13, 23, 15]
25 Ops.INDEX dtypes.f32 [21, 24]
26 Ops.WHERE dtypes.f32 [10, 25, 19]
27 Ops.MUL dtypes.f32 [20, 26]
28 Ops.REDUCE dtypes.f32 [27, 9] (Ops.ADD, 0)
29 Ops.STORE dtypes.void [7, 28]
30 Ops.END dtypes.void [29, 3, 5]
31 Ops.SINK dtypes.void [30] KernelInfo(name="r_4_4_8", applied_opts=(Opt(op=OptOps.PADTO, axis=2, arg=8),), optimized=True)

