# tolk D141 golden=7e6114185c3436a3e722d5c011374c95
0 Ops.CONST dtypes.weakint [] 1572864
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2048
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 768
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 24
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
18 Ops.CONST dtypes.weakint [] 32768
19 Ops.PARAM dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.MUL dtypes.weakint [3, 12]
21 Ops.CONST dtypes.weakint [] 1
22 Ops.RANGE dtypes.weakint [21] (AxisType.WEAK, 0)
23 Ops.MUL dtypes.weakint [22, 12]
24 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 5)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.ADD dtypes.weakint [20, 25]
27 Ops.INDEX dtypes.f32 [19, 26]
28 Ops.CONST dtypes.weakint [] 12288
29 Ops.PARAM dtypes.f32 [28] ParamArg(slot=2, dtype=dtypes.f32)
30 Ops.MUL dtypes.weakint [25, 4]
31 Ops.ADD dtypes.weakint [30, 15]
32 Ops.INDEX dtypes.f32 [29, 31]
33 Ops.MUL dtypes.f32 [27, 32]
34 Ops.REDUCE dtypes.f32 [33, 22, 24] (Ops.ADD, 0)
35 Ops.STORE dtypes.void [17, 34]
36 Ops.END dtypes.void [35, 3, 7, 10, 14]
37 Ops.SINK dtypes.void [36] KernelInfo(name="r_2048_24_16_2_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(0, AxisType.UPCAST))), optimized=True)

