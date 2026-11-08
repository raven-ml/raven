# tolk D89 D280 golden=47b065c4106be2769e219d2e14a35df7
0 Ops.CONST dtypes.weakint [] 128
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 8
9 Ops.RANGE dtypes.weakint [8] (AxisType.GLOBAL, 2)
10 Ops.MUL dtypes.weakint [9, 4]
11 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.AFTER dtypes.f32 [1, 7, 12]
14 Ops.CONST dtypes.weakint [] 32
15 Ops.MUL dtypes.weakint [7, 14]
16 Ops.ADD dtypes.weakint [15, 12]
17 Ops.INDEX dtypes.f32 [13, 16]
18 Ops.CONST dtypes.weakint [] 1024
19 Ops.PARAM dtypes.bf16 [18] ParamArg(slot=1, dtype=dtypes.bf16)
20 Ops.CONST dtypes.weakint [] 256
21 Ops.MUL dtypes.weakint [7, 20]
22 Ops.CONST dtypes.weakint [] 2
23 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [26, 14]
28 Ops.RANGE dtypes.weakint [14] (AxisType.LOCAL, 4)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [21, 29]
31 Ops.INDEX dtypes.bf16 [19, 30]
32 Ops.CAST dtypes.f32 [31] dtypes.f32
33 Ops.CONST dtypes.weakint [] 8192
34 Ops.PARAM dtypes.bf16 [33] ParamArg(slot=2, dtype=dtypes.bf16)
35 Ops.MUL dtypes.weakint [12, 20]
36 Ops.ADD dtypes.weakint [35, 29]
37 Ops.INDEX dtypes.bf16 [34, 36]
38 Ops.CAST dtypes.f32 [37] dtypes.f32
39 Ops.MUL dtypes.f32 [32, 38]
40 Ops.REDUCE dtypes.f32 [39, 23, 25, 28] (Ops.ADD, 0)
41 Ops.STORE dtypes.void [17, 40]
42 Ops.END dtypes.void [41, 3, 6, 9, 11]
43 Ops.SINK dtypes.void [42] KernelInfo(name="r_8_32_4_4_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

