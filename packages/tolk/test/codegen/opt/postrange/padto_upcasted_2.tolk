# tolk D280 golden=21070c07dead82efe643a1bbb4244c4f
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
5 Ops.AFTER dtypes.f32 [1, 3, 4]
6 Ops.MUL dtypes.weakint [3, 2]
7 Ops.ADD dtypes.weakint [6, 4]
8 Ops.INDEX dtypes.f32 [5, 7]
9 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
10 Ops.CONST dtypes.weakint [] 1
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 0)
12 Ops.MUL dtypes.weakint [11, 2]
13 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [6, 14]
16 Ops.INDEX dtypes.f32 [9, 15]
17 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
18 Ops.MUL dtypes.weakint [14, 2]
19 Ops.ADD dtypes.weakint [18, 4]
20 Ops.INDEX dtypes.f32 [17, 19]
21 Ops.MUL dtypes.f32 [16, 20]
22 Ops.REDUCE dtypes.f32 [21, 11, 13] (Ops.ADD, 0)
23 Ops.STORE dtypes.void [8, 22]
24 Ops.END dtypes.void [23, 3, 4]
25 Ops.SINK dtypes.void [24] KernelInfo(name="r_4_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(0, AxisType.UPCAST)),), optimized=True)

