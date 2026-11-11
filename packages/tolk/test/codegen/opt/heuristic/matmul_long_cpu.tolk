# tolk D141 golden=1c7faca5c0cdabd98a7316587ac910cd
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.CONST dtypes.weakint [] 2
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 2)
7 Ops.MUL dtypes.weakint [6, 5]
8 Ops.RANGE dtypes.weakint [5] (AxisType.UPCAST, 4)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [4, 13]
15 Ops.INDEX dtypes.f32 [1, 14]
16 Ops.CONST dtypes.weakint [] 65536
17 Ops.PARAM dtypes.f32 [16] ParamArg(slot=1, dtype=dtypes.f32)
18 Ops.CONST dtypes.weakint [] 1024
19 Ops.MUL dtypes.weakint [3, 18]
20 Ops.CONST dtypes.weakint [] 256
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
22 Ops.CONST dtypes.weakint [] 4
23 Ops.MUL dtypes.weakint [21, 22]
24 Ops.RANGE dtypes.weakint [22] (AxisType.UPCAST, 5)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.ADD dtypes.weakint [19, 25]
27 Ops.INDEX dtypes.f32 [17, 26]
28 Ops.PARAM dtypes.f32 [16] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [25, 2]
30 Ops.ADD dtypes.weakint [29, 13]
31 Ops.INDEX dtypes.f32 [28, 30]
32 Ops.MUL dtypes.f32 [27, 31]
33 Ops.REDUCE dtypes.f32 [32, 21, 24] (Ops.ADD, 0)
34 Ops.STORE dtypes.void [15, 33]
35 Ops.END dtypes.void [34, 3, 6, 8, 12]
36 Ops.SINK dtypes.void [35] KernelInfo(name="r_64_2_16_2_256_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

