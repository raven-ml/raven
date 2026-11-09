# tolk D89 golden=78a20cb1793d404b586356d141555619
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 2880
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 360
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0)
17 Ops.CONST dtypes.weakint [] 8
18 Ops.MUL dtypes.weakint [16, 17]
19 Ops.RANGE dtypes.weakint [17] (AxisType.LOCAL, 4)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.INDEX dtypes.f32 [14, 20]
22 Ops.PARAM dtypes.f32 [13] ParamArg(slot=3, dtype=dtypes.f32)
23 Ops.INDEX dtypes.f32 [22, 20]
24 Ops.MUL dtypes.f32 [21, 23]
25 Ops.CONST dtypes.weakint [] 11796480
26 Ops.PARAM dtypes.bf16 [25] ParamArg(slot=4, dtype=dtypes.bf16)
27 Ops.MUL dtypes.weakint [20, 0]
28 Ops.ADD dtypes.weakint [27, 11]
29 Ops.INDEX dtypes.bf16 [26, 28]
30 Ops.CAST dtypes.f32 [29] dtypes.f32
31 Ops.MUL dtypes.f32 [24, 30]
32 Ops.REDUCE dtypes.f32 [31, 16, 19] (Ops.ADD, 0)
33 Ops.CONST dtypes.weakint [] 1
34 Ops.PARAM dtypes.f32 [33] ParamArg(slot=2, dtype=dtypes.f32)
35 Ops.CONST dtypes.weakint [] 0
36 Ops.INDEX dtypes.f32 [34, 35]
37 Ops.RECIP dtypes.f32 [36]
38 Ops.MUL dtypes.f32 [32, 37]
39 Ops.STORE dtypes.void [12, 38]
40 Ops.END dtypes.void [39, 3, 6, 10]
41 Ops.SINK dtypes.void [40] KernelInfo(name="r_64_32_8_2_360", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(8, AxisType.LOCAL))), optimized=True)

