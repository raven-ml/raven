# tolk D89 golden=be86af26366af34c14abe3975d66ab62
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 16
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 4096
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 64
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0)
17 Ops.MUL dtypes.weakint [16, 15]
18 Ops.RANGE dtypes.weakint [15] (AxisType.LOCAL, 4)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.INDEX dtypes.f32 [14, 19]
21 Ops.PARAM dtypes.f32 [13] ParamArg(slot=3, dtype=dtypes.f32)
22 Ops.INDEX dtypes.f32 [21, 19]
23 Ops.MUL dtypes.f32 [20, 22]
24 Ops.CONST dtypes.weakint [] 4194304
25 Ops.PARAM dtypes.f32 [24] ParamArg(slot=4, dtype=dtypes.f32)
26 Ops.MUL dtypes.weakint [19, 0]
27 Ops.ADD dtypes.weakint [26, 11]
28 Ops.INDEX dtypes.f32 [25, 27]
29 Ops.MUL dtypes.f32 [23, 28]
30 Ops.REDUCE dtypes.f32 [29, 16, 18] (Ops.ADD, 0)
31 Ops.CONST dtypes.weakint [] 1
32 Ops.PARAM dtypes.f32 [31] ParamArg(slot=2, dtype=dtypes.f32)
33 Ops.CONST dtypes.weakint [] 0
34 Ops.INDEX dtypes.f32 [32, 33]
35 Ops.RECIP dtypes.f32 [34]
36 Ops.MUL dtypes.f32 [30, 35]
37 Ops.STORE dtypes.void [12, 36]
38 Ops.END dtypes.void [37, 3, 6, 10]
39 Ops.SINK dtypes.void [38] KernelInfo(name="r_32_16_64_2_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(64, AxisType.LOCAL))), optimized=True)

