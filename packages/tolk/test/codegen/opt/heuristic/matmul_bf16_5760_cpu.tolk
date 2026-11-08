# tolk D141 D280 golden=11848f6fbcc5e8e63c4c863d8eaded6e
0 Ops.CONST dtypes.weakint [] 2949120
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 512
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 180
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
6 Ops.CONST dtypes.weakint [] 2
7 Ops.MUL dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [6] (AxisType.UPCAST, 4)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 3, 13]
15 Ops.CONST dtypes.weakint [] 5760
16 Ops.MUL dtypes.weakint [3, 15]
17 Ops.ADD dtypes.weakint [16, 13]
18 Ops.INDEX dtypes.f32 [14, 17]
19 Ops.CONST dtypes.weakint [] 1474560
20 Ops.PARAM dtypes.bf16 [19] ParamArg(slot=1, dtype=dtypes.bf16)
21 Ops.CONST dtypes.weakint [] 2880
22 Ops.MUL dtypes.weakint [3, 21]
23 Ops.RANGE dtypes.weakint [21] (AxisType.WEAK, 0)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.INDEX dtypes.bf16 [20, 24]
26 Ops.CAST dtypes.f32 [25] dtypes.f32
27 Ops.CONST dtypes.weakint [] 16588800
28 Ops.PARAM dtypes.bf16 [27] ParamArg(slot=2, dtype=dtypes.bf16)
29 Ops.MUL dtypes.weakint [23, 15]
30 Ops.ADD dtypes.weakint [29, 13]
31 Ops.INDEX dtypes.bf16 [28, 30]
32 Ops.CAST dtypes.f32 [31] dtypes.f32
33 Ops.MUL dtypes.f32 [26, 32]
34 Ops.REDUCE dtypes.f32 [33, 23] (Ops.ADD, 0)
35 Ops.STORE dtypes.void [18, 34]
36 Ops.END dtypes.void [35, 3, 5, 8, 12]
37 Ops.SINK dtypes.void [36] KernelInfo(name="r_512_180_16_2_2880", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST))), optimized=True)

