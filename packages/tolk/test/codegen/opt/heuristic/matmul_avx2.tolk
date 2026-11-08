# tolk D141 D280 golden=b390b3417da055963bb2a363936c9566
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 8
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
10 Ops.CONST dtypes.weakint [] 16
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 7, 13]
15 Ops.CONST dtypes.weakint [] 128
16 Ops.MUL dtypes.weakint [7, 15]
17 Ops.ADD dtypes.weakint [16, 13]
18 Ops.INDEX dtypes.f32 [14, 17]
19 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
21 Ops.MUL dtypes.weakint [20, 4]
22 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.ADD dtypes.weakint [16, 23]
25 Ops.INDEX dtypes.f32 [19, 24]
26 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
27 Ops.MUL dtypes.weakint [23, 15]
28 Ops.ADD dtypes.weakint [27, 13]
29 Ops.INDEX dtypes.f32 [26, 28]
30 Ops.MUL dtypes.f32 [25, 29]
31 Ops.REDUCE dtypes.f32 [30, 20, 22] (Ops.ADD, 0)
32 Ops.STORE dtypes.void [18, 31]
33 Ops.END dtypes.void [32, 3, 6, 9, 12]
34 Ops.SINK dtypes.void [33] KernelInfo(name="r_32_8_16_4_32_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

