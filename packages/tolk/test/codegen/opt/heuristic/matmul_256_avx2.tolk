# tolk D141 golden=7a0ee12fc1222201d637badc541d9ce3
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 256
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.MUL dtypes.weakint [11, 10]
13 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [9, 14]
16 Ops.INDEX dtypes.f32 [1, 15]
17 Ops.CONST dtypes.weakint [] 16384
18 Ops.PARAM dtypes.f32 [17] ParamArg(slot=1, dtype=dtypes.f32)
19 Ops.MUL dtypes.weakint [7, 2]
20 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 0)
21 Ops.MUL dtypes.weakint [20, 4]
22 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.ADD dtypes.weakint [19, 23]
25 Ops.INDEX dtypes.f32 [18, 24]
26 Ops.PARAM dtypes.f32 [17] ParamArg(slot=2, dtype=dtypes.f32)
27 Ops.MUL dtypes.weakint [23, 8]
28 Ops.ADD dtypes.weakint [27, 14]
29 Ops.INDEX dtypes.f32 [26, 28]
30 Ops.MUL dtypes.f32 [25, 29]
31 Ops.REDUCE dtypes.f32 [30, 20, 22] (Ops.ADD, 0)
32 Ops.STORE dtypes.void [16, 31]
33 Ops.END dtypes.void [32, 3, 6, 11, 13]
34 Ops.SINK dtypes.void [33] KernelInfo(name="r_64_16_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

