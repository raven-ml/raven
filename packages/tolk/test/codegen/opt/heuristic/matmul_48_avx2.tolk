# tolk D141 golden=ee1ac811d9a60081bd7a0800e3096a2e
0 Ops.CONST dtypes.weakint [] 49152
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 3
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 1024
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 64
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.MUL dtypes.weakint [11, 2]
13 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [9, 14]
16 Ops.INDEX dtypes.f32 [1, 15]
17 Ops.CONST dtypes.weakint [] 3072
18 Ops.PARAM dtypes.f32 [17] ParamArg(slot=1, dtype=dtypes.f32)
19 Ops.MUL dtypes.weakint [7, 10]
20 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
21 Ops.CONST dtypes.weakint [] 4
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.ADD dtypes.weakint [19, 24]
26 Ops.INDEX dtypes.f32 [18, 25]
27 Ops.CONST dtypes.weakint [] 65536
28 Ops.PARAM dtypes.f32 [27] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [24, 8]
30 Ops.ADD dtypes.weakint [29, 14]
31 Ops.INDEX dtypes.f32 [28, 30]
32 Ops.MUL dtypes.f32 [26, 31]
33 Ops.REDUCE dtypes.f32 [32, 20, 23] (Ops.ADD, 0)
34 Ops.STORE dtypes.void [16, 33]
35 Ops.END dtypes.void [34, 3, 6, 11, 13]
36 Ops.SINK dtypes.void [35] KernelInfo(name="r_16_64_16_3_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(3, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

