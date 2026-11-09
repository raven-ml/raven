# tolk D141 golden=ed10a0202018106066c57a7a7058dd4c
0 Ops.CONST dtypes.weakint [] 128
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 8
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.MUL dtypes.weakint [11, 4]
13 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [9, 14]
16 Ops.INDEX dtypes.f32 [1, 15]
17 Ops.CONST dtypes.weakint [] 1024
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
29 Ops.CONST dtypes.weakint [] 8192
30 Ops.PARAM dtypes.bf16 [29] ParamArg(slot=2, dtype=dtypes.bf16)
31 Ops.MUL dtypes.weakint [14, 19]
32 Ops.ADD dtypes.weakint [31, 25]
33 Ops.INDEX dtypes.bf16 [30, 32]
34 Ops.CAST dtypes.f32 [33] dtypes.f32
35 Ops.MUL dtypes.f32 [28, 34]
36 Ops.REDUCE dtypes.f32 [35, 22, 24] (Ops.ADD, 0)
37 Ops.STORE dtypes.void [16, 36]
38 Ops.END dtypes.void [37, 3, 6, 11, 13]
39 Ops.SINK dtypes.void [38] KernelInfo(name="r_8_4_4_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(4, AxisType.UPCAST))), optimized=True)

