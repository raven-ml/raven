# tolk D141 golden=1c7faca5c0cdabd98a7316587ac910cd
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 64
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
11 Ops.MUL dtypes.weakint [10, 2]
12 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [9, 13]
15 Ops.INDEX dtypes.f32 [1, 14]
16 Ops.CONST dtypes.weakint [] 65536
17 Ops.PARAM dtypes.f32 [16] ParamArg(slot=1, dtype=dtypes.f32)
18 Ops.CONST dtypes.weakint [] 1024
19 Ops.MUL dtypes.weakint [7, 18]
20 Ops.CONST dtypes.weakint [] 256
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.ADD dtypes.weakint [19, 24]
26 Ops.INDEX dtypes.f32 [17, 25]
27 Ops.PARAM dtypes.f32 [16] ParamArg(slot=2, dtype=dtypes.f32)
28 Ops.MUL dtypes.weakint [24, 8]
29 Ops.ADD dtypes.weakint [28, 13]
30 Ops.INDEX dtypes.f32 [27, 29]
31 Ops.MUL dtypes.f32 [26, 30]
32 Ops.REDUCE dtypes.f32 [31, 21, 23] (Ops.ADD, 0)
33 Ops.STORE dtypes.void [15, 32]
34 Ops.END dtypes.void [33, 3, 6, 10, 12]
35 Ops.SINK dtypes.void [34] KernelInfo(name="r_16_4_16_4_256_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

