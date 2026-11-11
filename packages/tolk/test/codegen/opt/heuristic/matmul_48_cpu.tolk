# tolk D141 golden=ee1ac811d9a60081bd7a0800e3096a2e
0 Ops.CONST dtypes.weakint [] 49152
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 48
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 1024
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 2)
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 4)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 16
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [5, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 3072
19 Ops.PARAM dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.CONST dtypes.weakint [] 64
21 Ops.MUL dtypes.weakint [3, 20]
22 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
23 Ops.CONST dtypes.weakint [] 4
24 Ops.MUL dtypes.weakint [22, 23]
25 Ops.RANGE dtypes.weakint [23] (AxisType.UPCAST, 5)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.ADD dtypes.weakint [21, 26]
28 Ops.INDEX dtypes.f32 [19, 27]
29 Ops.CONST dtypes.weakint [] 65536
30 Ops.PARAM dtypes.f32 [29] ParamArg(slot=2, dtype=dtypes.f32)
31 Ops.MUL dtypes.weakint [26, 4]
32 Ops.ADD dtypes.weakint [31, 15]
33 Ops.INDEX dtypes.f32 [30, 32]
34 Ops.MUL dtypes.f32 [28, 33]
35 Ops.REDUCE dtypes.f32 [34, 22, 25] (Ops.ADD, 0)
36 Ops.STORE dtypes.void [17, 35]
37 Ops.END dtypes.void [36, 3, 7, 10, 14]
38 Ops.SINK dtypes.void [37] KernelInfo(name="r_48_32_16_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

