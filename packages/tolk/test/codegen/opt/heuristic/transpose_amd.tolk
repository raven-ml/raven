# tolk D280 golden=89038363c28a9e6b9e6795e7f385c975
0 Ops.CONST dtypes.weakint [] 8192
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 0)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 1
9 Ops.RANGE dtypes.weakint [8] (AxisType.GLOBAL, 1)
10 Ops.MUL dtypes.weakint [9, 2]
11 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 4)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.CONST dtypes.weakint [] 4
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 2)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.AFTER dtypes.f32 [1, 7, 16]
18 Ops.CONST dtypes.weakint [] 64
19 Ops.MUL dtypes.weakint [7, 18]
20 Ops.ADD dtypes.weakint [19, 16]
21 Ops.INDEX dtypes.f32 [17, 20]
22 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
23 Ops.CONST dtypes.weakint [] 128
24 Ops.MUL dtypes.weakint [16, 23]
25 Ops.ADD dtypes.weakint [24, 7]
26 Ops.INDEX dtypes.f32 [22, 25]
27 Ops.STORE dtypes.void [21, 26]
28 Ops.END dtypes.void [27, 3, 6, 9, 11, 15]
29 Ops.SINK dtypes.void [28] KernelInfo(name="E_16_8_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL))), optimized=True)

