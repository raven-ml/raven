# tolk D280 golden=336c863711dd4a781043ea71b232822a
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 3)
5 Ops.AFTER dtypes.f32 [1, 3, 4]
6 Ops.MUL dtypes.weakint [3, 2]
7 Ops.ADD dtypes.weakint [6, 4]
8 Ops.INDEX dtypes.f32 [5, 7]
9 Ops.CONST dtypes.weakint [] 1048576
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 128
12 Ops.MUL dtypes.weakint [4, 11]
13 Ops.CONST dtypes.weakint [] 2
14 Ops.RANGE dtypes.weakint [13] (AxisType.LOCAL, 5)
15 Ops.MUL dtypes.weakint [14, 0]
16 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [12, 17]
19 Ops.CONST dtypes.weakint [] 16
20 Ops.RANGE dtypes.weakint [19] (AxisType.LOCAL, 4)
21 Ops.MUL dtypes.weakint [20, 2]
22 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.CONST dtypes.weakint [] 1024
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.ADD dtypes.weakint [18, 25]
27 Ops.CONST dtypes.weakint [] 131072
28 Ops.MUL dtypes.weakint [3, 27]
29 Ops.ADD dtypes.weakint [26, 28]
30 Ops.INDEX dtypes.f32 [10, 29]
31 Ops.REDUCE dtypes.f32 [30, 20, 22, 14, 16] (Ops.ADD, 0)
32 Ops.STORE dtypes.void [8, 31]
33 Ops.END dtypes.void [32, 3, 4]
34 Ops.SINK dtypes.void [33] KernelInfo(name="r_8_8_16_2_8_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.LOCAL, True)), Opt(op=OptOps.SPLIT, axis=4, arg=(2, AxisType.LOCAL, True))), optimized=True)

