# tolk D141 golden=d4f28d986a84e38e4e5c368953741ec0
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
17 Ops.PARAM dtypes.bf16 [16] ParamArg(slot=1, dtype=dtypes.bf16)
18 Ops.CONST dtypes.weakint [] 1024
19 Ops.MUL dtypes.weakint [3, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.WEAK, 0)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.INDEX dtypes.bf16 [17, 21]
23 Ops.CAST dtypes.f32 [22] dtypes.f32
24 Ops.PARAM dtypes.bf16 [16] ParamArg(slot=2, dtype=dtypes.bf16)
25 Ops.MUL dtypes.weakint [20, 2]
26 Ops.ADD dtypes.weakint [25, 13]
27 Ops.INDEX dtypes.bf16 [24, 26]
28 Ops.CAST dtypes.f32 [27] dtypes.f32
29 Ops.MUL dtypes.f32 [23, 28]
30 Ops.REDUCE dtypes.f32 [29, 20] (Ops.ADD, 0)
31 Ops.STORE dtypes.void [15, 30]
32 Ops.END dtypes.void [31, 3, 6, 8, 12]
33 Ops.SINK dtypes.void [32] KernelInfo(name="r_64_2_16_2_1024", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST))), optimized=True)

