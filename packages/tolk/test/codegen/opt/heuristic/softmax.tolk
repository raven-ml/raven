# tolk D280 golden=bf742786779b4d5056c6c52047806bfb
0 Ops.CONST dtypes.weakint [] 8192
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 256
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
6 Ops.AFTER dtypes.f32 [1, 3, 5]
7 Ops.MUL dtypes.weakint [3, 4]
8 Ops.ADD dtypes.weakint [7, 5]
9 Ops.INDEX dtypes.f32 [6, 8]
10 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.INDEX dtypes.f32 [10, 8]
12 Ops.PARAM dtypes.f32 [2] ParamArg(slot=2, dtype=dtypes.f32)
13 Ops.INDEX dtypes.f32 [12, 3]
14 Ops.CONST dtypes.weakfloat [] -1.0
15 Ops.MUL dtypes.f32 [13, 14]
16 Ops.ADD dtypes.f32 [11, 15]
17 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
18 Ops.MUL dtypes.f32 [16, 17]
19 Ops.EXP2 dtypes.f32 [18]
20 Ops.PARAM dtypes.f32 [2] ParamArg(slot=3, dtype=dtypes.f32)
21 Ops.INDEX dtypes.f32 [20, 3]
22 Ops.RECIP dtypes.f32 [21]
23 Ops.MUL dtypes.f32 [19, 22]
24 Ops.STORE dtypes.void [9, 23]
25 Ops.END dtypes.void [24, 3, 5]
26 Ops.SINK dtypes.void [25] KernelInfo()

