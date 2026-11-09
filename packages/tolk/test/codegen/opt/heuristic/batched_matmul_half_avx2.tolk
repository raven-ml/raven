# tolk D141 golden=2f84d53f0125d78ba60a4be6ccd75359
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 64
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 3)
11 Ops.MUL dtypes.weakint [10, 2]
12 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 4)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [9, 13]
15 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
16 Ops.CONST dtypes.weakint [] 4096
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [14, 17]
19 Ops.INDEX dtypes.f16 [1, 18]
20 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
21 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
22 Ops.ADD dtypes.weakint [9, 21]
23 Ops.ADD dtypes.weakint [22, 17]
24 Ops.INDEX dtypes.f16 [20, 23]
25 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
26 Ops.MUL dtypes.weakint [21, 8]
27 Ops.ADD dtypes.weakint [26, 13]
28 Ops.ADD dtypes.weakint [27, 17]
29 Ops.INDEX dtypes.f16 [25, 28]
30 Ops.MUL dtypes.f16 [24, 29]
31 Ops.CAST dtypes.f32 [30] dtypes.f32
32 Ops.REDUCE dtypes.f32 [31, 21] (Ops.ADD, 0)
33 Ops.CAST dtypes.f16 [32] dtypes.f16
34 Ops.STORE dtypes.void [19, 33]
35 Ops.END dtypes.void [34, 15, 3, 6, 10, 12]
36 Ops.SINK dtypes.void [35] KernelInfo(name="r_4_16_4_16_4_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST))), optimized=True)

