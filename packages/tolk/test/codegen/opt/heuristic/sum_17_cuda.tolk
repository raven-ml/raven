# tolk D280 golden=03b1c82107c6fac14812eaa2802dbf03
0 Ops.CONST dtypes.weakint [] 512
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 32
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.AFTER dtypes.f32 [1, 7]
9 Ops.INDEX dtypes.f32 [8, 7]
10 Ops.CONST dtypes.weakint [] 8704
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 17
13 Ops.MUL dtypes.weakint [7, 12]
14 Ops.CONST dtypes.weakint [] 1
15 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 0)
16 Ops.MUL dtypes.weakint [15, 12]
17 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 2)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.ADD dtypes.weakint [13, 18]
20 Ops.INDEX dtypes.f32 [11, 19]
21 Ops.REDUCE dtypes.f32 [20, 15, 17] (Ops.ADD, 0)
22 Ops.STORE dtypes.void [9, 21]
23 Ops.END dtypes.void [22, 3, 6]
24 Ops.SINK dtypes.void [23] KernelInfo(name="r_16_32_17", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL))), optimized=True)

