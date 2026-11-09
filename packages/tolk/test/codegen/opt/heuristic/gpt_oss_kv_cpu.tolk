# tolk D141 golden=60380cc42e522833b063973658573ce7
0 Ops.CONST dtypes.weakint [] 512
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 2]
9 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 2)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.INDEX dtypes.f32 [1, 10]
12 Ops.CONST dtypes.weakint [] 2880
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
15 Ops.INDEX dtypes.f32 [13, 14]
16 Ops.PARAM dtypes.f32 [12] ParamArg(slot=3, dtype=dtypes.f32)
17 Ops.INDEX dtypes.f32 [16, 14]
18 Ops.MUL dtypes.f32 [15, 17]
19 Ops.CONST dtypes.weakint [] 1474560
20 Ops.PARAM dtypes.bf16 [19] ParamArg(slot=4, dtype=dtypes.bf16)
21 Ops.MUL dtypes.weakint [14, 0]
22 Ops.ADD dtypes.weakint [21, 10]
23 Ops.INDEX dtypes.bf16 [20, 22]
24 Ops.CAST dtypes.f32 [23] dtypes.f32
25 Ops.MUL dtypes.f32 [18, 24]
26 Ops.REDUCE dtypes.f32 [25, 14] (Ops.ADD, 0)
27 Ops.CONST dtypes.weakint [] 1
28 Ops.PARAM dtypes.f32 [27] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.CONST dtypes.weakint [] 0
30 Ops.INDEX dtypes.f32 [28, 29]
31 Ops.RECIP dtypes.f32 [30]
32 Ops.MUL dtypes.f32 [26, 31]
33 Ops.STORE dtypes.void [11, 32]
34 Ops.END dtypes.void [33, 3, 6, 9]
35 Ops.SINK dtypes.void [34] KernelInfo(name="r_16_16_2_2880", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST))), optimized=True)

