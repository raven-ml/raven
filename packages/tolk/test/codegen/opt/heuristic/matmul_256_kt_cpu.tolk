# tolk D188 golden=9dfb92dab3d1064578c0718a75b32be7
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 256
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 64
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 16384
19 Ops.PARAM dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.MUL dtypes.weakint [7, 10]
21 Ops.CONST dtypes.weakint [] 16
22 Ops.RANGE dtypes.weakint [21] (AxisType.WEAK, 0)
23 Ops.MUL dtypes.weakint [22, 12]
24 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 5)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.ADD dtypes.weakint [20, 25]
27 Ops.INDEX dtypes.f32 [19, 26]
28 Ops.PARAM dtypes.f32 [18] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [15, 10]
30 Ops.ADD dtypes.weakint [29, 25]
31 Ops.INDEX dtypes.f32 [28, 30]
32 Ops.MUL dtypes.f32 [27, 31]
33 Ops.REDUCE dtypes.f32 [32, 22, 24] (Ops.ADD, 0)
34 Ops.STORE dtypes.void [17, 33]
35 Ops.END dtypes.void [34, 3, 6, 11, 14]
36 Ops.SINK dtypes.void [35] KernelInfo(name="r_32_64_4_8_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

