# tolk D141 golden=3a307ef3666545981fcb32ce562762b6
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
16 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
17 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
18 Ops.ADD dtypes.weakint [4, 17]
19 Ops.INDEX dtypes.f32 [16, 18]
20 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=2, dtype=dtypes.bf16)
21 Ops.MUL dtypes.weakint [17, 2]
22 Ops.ADD dtypes.weakint [21, 13]
23 Ops.INDEX dtypes.bf16 [20, 22]
24 Ops.CAST dtypes.f32 [23] dtypes.f32
25 Ops.MUL dtypes.f32 [19, 24]
26 Ops.REDUCE dtypes.f32 [25, 17] (Ops.ADD, 0)
27 Ops.STORE dtypes.void [15, 26]
28 Ops.END dtypes.void [27, 3, 6, 8, 12]
29 Ops.SINK dtypes.void [28] KernelInfo(name="r_64_2_16_2_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST))), optimized=True)

