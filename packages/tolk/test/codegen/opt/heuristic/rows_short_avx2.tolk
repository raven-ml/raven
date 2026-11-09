# tolk D141 golden=e473d117ed4e00c9ca1b5ae30d4ce398
0 Ops.CONST dtypes.weakint [] 512
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 4)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.CONST dtypes.weakint [] 32
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.CONST dtypes.weakint [] 8
10 Ops.RANGE dtypes.weakint [9] (AxisType.WEAK, 2)
11 Ops.MUL dtypes.weakint [10, 2]
12 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [8, 13]
15 Ops.INDEX dtypes.f32 [1, 14]
16 Ops.CONST dtypes.weakint [] 1024
17 Ops.PARAM dtypes.bf16 [16] ParamArg(slot=1, dtype=dtypes.bf16)
18 Ops.CONST dtypes.weakint [] 64
19 Ops.MUL dtypes.weakint [6, 18]
20 Ops.CONST dtypes.weakint [] 16
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
22 Ops.MUL dtypes.weakint [21, 2]
23 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.ADD dtypes.weakint [19, 24]
26 Ops.INDEX dtypes.bf16 [17, 25]
27 Ops.CAST dtypes.f32 [26] dtypes.f32
28 Ops.CONST dtypes.weakint [] 2048
29 Ops.PARAM dtypes.bf16 [28] ParamArg(slot=2, dtype=dtypes.bf16)
30 Ops.MUL dtypes.weakint [13, 18]
31 Ops.ADD dtypes.weakint [30, 24]
32 Ops.INDEX dtypes.bf16 [29, 31]
33 Ops.CAST dtypes.f32 [32] dtypes.f32
34 Ops.MUL dtypes.f32 [27, 33]
35 Ops.REDUCE dtypes.f32 [34, 21, 23] (Ops.ADD, 0)
36 Ops.STORE dtypes.void [15, 35]
37 Ops.END dtypes.void [36, 3, 5, 10, 12]
38 Ops.SINK dtypes.void [37] KernelInfo(name="r_4_8_4_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

