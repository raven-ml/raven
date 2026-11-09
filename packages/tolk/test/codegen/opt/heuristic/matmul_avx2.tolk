# tolk D141 golden=b390b3417da055963bb2a363936c9566
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 128
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 8
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.CONST dtypes.weakint [] 16
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
19 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.ADD dtypes.weakint [9, 22]
24 Ops.INDEX dtypes.f32 [18, 23]
25 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
26 Ops.MUL dtypes.weakint [22, 8]
27 Ops.ADD dtypes.weakint [26, 15]
28 Ops.INDEX dtypes.f32 [25, 27]
29 Ops.MUL dtypes.f32 [24, 28]
30 Ops.REDUCE dtypes.f32 [29, 19, 21] (Ops.ADD, 0)
31 Ops.STORE dtypes.void [17, 30]
32 Ops.END dtypes.void [31, 3, 6, 11, 14]
33 Ops.SINK dtypes.void [32] KernelInfo(name="r_32_8_16_4_32_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

