# tolk D141 D280 golden=2f3b0188b0f73c399fe9e988ec2a0bc2
0 Ops.CONST dtypes.weakint [] 8192
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 16
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
6 Ops.MUL dtypes.weakint [5, 4]
7 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
8 Ops.ADD dtypes.weakint [6, 7]
9 Ops.AFTER dtypes.f32 [1, 3, 8]
10 Ops.CONST dtypes.weakint [] 256
11 Ops.MUL dtypes.weakint [3, 10]
12 Ops.ADD dtypes.weakint [11, 8]
13 Ops.INDEX dtypes.f32 [9, 12]
14 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.INDEX dtypes.f32 [14, 12]
16 Ops.PARAM dtypes.f32 [2] ParamArg(slot=2, dtype=dtypes.f32)
17 Ops.INDEX dtypes.f32 [16, 3]
18 Ops.CONST dtypes.weakfloat [] -1.0
19 Ops.MUL dtypes.f32 [17, 18]
20 Ops.ADD dtypes.f32 [15, 19]
21 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
22 Ops.MUL dtypes.f32 [20, 21]
23 Ops.EXP2 dtypes.f32 [22]
24 Ops.PARAM dtypes.f32 [2] ParamArg(slot=3, dtype=dtypes.f32)
25 Ops.INDEX dtypes.f32 [24, 3]
26 Ops.RECIP dtypes.f32 [25]
27 Ops.MUL dtypes.f32 [23, 26]
28 Ops.STORE dtypes.void [13, 27]
29 Ops.END dtypes.void [28, 3, 5, 7]
30 Ops.SINK dtypes.void [29] KernelInfo(name="E_32_16_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)),), optimized=True)

