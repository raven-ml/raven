# tolk D89 D280 golden=aecb3601736716e478f807bcc04a236f
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.RANGE dtypes.weakint [8] (AxisType.GLOBAL, 2)
10 Ops.CONST dtypes.weakint [] 16
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 7, 13]
15 Ops.CONST dtypes.weakint [] 32
16 Ops.MUL dtypes.weakint [7, 15]
17 Ops.ADD dtypes.weakint [16, 13]
18 Ops.INDEX dtypes.f32 [14, 17]
19 Ops.CONST dtypes.weakint [] 8192
20 Ops.PARAM dtypes.bf16 [19] ParamArg(slot=1, dtype=dtypes.bf16)
21 Ops.CONST dtypes.weakint [] 256
22 Ops.MUL dtypes.weakint [7, 21]
23 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [26, 15]
28 Ops.RANGE dtypes.weakint [15] (AxisType.LOCAL, 4)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [22, 29]
31 Ops.INDEX dtypes.bf16 [20, 30]
32 Ops.CAST dtypes.f32 [31] dtypes.f32
33 Ops.PARAM dtypes.bf16 [19] ParamArg(slot=2, dtype=dtypes.bf16)
34 Ops.MUL dtypes.weakint [13, 21]
35 Ops.ADD dtypes.weakint [34, 29]
36 Ops.INDEX dtypes.bf16 [33, 35]
37 Ops.CAST dtypes.f32 [36] dtypes.f32
38 Ops.MUL dtypes.f32 [32, 37]
39 Ops.REDUCE dtypes.f32 [38, 23, 25, 28] (Ops.ADD, 0)
40 Ops.STORE dtypes.void [18, 39]
41 Ops.END dtypes.void [40, 3, 6, 9, 12]
42 Ops.SINK dtypes.void [41] KernelInfo(name="r_8_2_32_16_4_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(4, AxisType.UPCAST))), optimized=True)

