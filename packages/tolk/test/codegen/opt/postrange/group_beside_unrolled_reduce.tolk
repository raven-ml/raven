# tolk D280 golden=b32e3b3c847dd6b0f9729a30b680e559
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.AFTER dtypes.f32 [1, 7]
9 Ops.INDEX dtypes.f32 [8, 7]
10 Ops.CONST dtypes.weakint [] 4096
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.MUL dtypes.weakint [7, 0]
13 Ops.CONST dtypes.weakint [] 1
14 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0)
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 4)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.MUL dtypes.weakint [17, 2]
19 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.ADD dtypes.weakint [12, 20]
22 Ops.INDEX dtypes.f32 [11, 21]
23 Ops.REDUCE dtypes.f32 [22, 14, 16, 19] (Ops.ADD, 0)
24 Ops.STORE dtypes.void [9, 23]
25 Ops.END dtypes.void [24, 3, 6]
26 Ops.SINK dtypes.void [25] KernelInfo(name="r_32_2_2_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(32, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.LOCAL))), optimized=True)

