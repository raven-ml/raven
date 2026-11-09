# tolk D89 golden=4449f1eeb22b36de3a46f46cdf6ae4a5
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 4096
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 128
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0)
17 Ops.MUL dtypes.weakint [16, 8]
18 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 4)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.INDEX dtypes.f32 [14, 19]
21 Ops.CONST dtypes.weakint [] 4194304
22 Ops.PARAM dtypes.f32 [21] ParamArg(slot=2, dtype=dtypes.f32)
23 Ops.MUL dtypes.weakint [19, 0]
24 Ops.ADD dtypes.weakint [23, 11]
25 Ops.INDEX dtypes.f32 [22, 24]
26 Ops.MUL dtypes.f32 [20, 25]
27 Ops.REDUCE dtypes.f32 [26, 16, 18] (Ops.ADD, 0)
28 Ops.STORE dtypes.void [12, 27]
29 Ops.END dtypes.void [28, 3, 6, 10]
30 Ops.SINK dtypes.void [29] KernelInfo(name="r_16_32_32_2_128", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(32, AxisType.LOCAL))), optimized=True)

