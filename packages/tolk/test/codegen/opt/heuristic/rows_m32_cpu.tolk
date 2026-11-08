# tolk D141 D280 golden=bdde997f8f20ebe14350d2ee0647d74a
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
9 Ops.MUL dtypes.weakint [8, 4]
10 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.AFTER dtypes.f32 [1, 7, 11]
13 Ops.CONST dtypes.weakint [] 32
14 Ops.MUL dtypes.weakint [7, 13]
15 Ops.ADD dtypes.weakint [14, 11]
16 Ops.INDEX dtypes.f32 [12, 15]
17 Ops.CONST dtypes.weakint [] 8192
18 Ops.PARAM dtypes.bf16 [17] ParamArg(slot=1, dtype=dtypes.bf16)
19 Ops.CONST dtypes.weakint [] 256
20 Ops.MUL dtypes.weakint [7, 19]
21 Ops.CONST dtypes.weakint [] 64
22 Ops.RANGE dtypes.weakint [21] (AxisType.WEAK, 0)
23 Ops.MUL dtypes.weakint [22, 4]
24 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.ADD dtypes.weakint [20, 25]
27 Ops.INDEX dtypes.bf16 [18, 26]
28 Ops.CAST dtypes.f32 [27] dtypes.f32
29 Ops.PARAM dtypes.bf16 [17] ParamArg(slot=2, dtype=dtypes.bf16)
30 Ops.MUL dtypes.weakint [11, 19]
31 Ops.ADD dtypes.weakint [30, 25]
32 Ops.INDEX dtypes.bf16 [29, 31]
33 Ops.CAST dtypes.f32 [32] dtypes.f32
34 Ops.MUL dtypes.f32 [28, 33]
35 Ops.REDUCE dtypes.f32 [34, 22, 24] (Ops.ADD, 0)
36 Ops.STORE dtypes.void [16, 35]
37 Ops.END dtypes.void [36, 3, 6, 8, 10]
38 Ops.SINK dtypes.void [37] KernelInfo(name="r_8_8_4_4_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

