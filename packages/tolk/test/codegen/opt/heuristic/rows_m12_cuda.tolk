# tolk D89 golden=09e7a74b7ceafbb255c44d6ce9a0cdb5
0 Ops.CONST dtypes.weakint [] 384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 12
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 8
11 Ops.RANGE dtypes.weakint [10] (AxisType.GLOBAL, 2)
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 5)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 3072
19 Ops.PARAM dtypes.bf16 [18] ParamArg(slot=1, dtype=dtypes.bf16)
20 Ops.CONST dtypes.weakint [] 256
21 Ops.MUL dtypes.weakint [7, 20]
22 Ops.CONST dtypes.weakint [] 2
23 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0)
24 Ops.MUL dtypes.weakint [23, 12]
25 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 6)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [26, 8]
28 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 4)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [21, 29]
31 Ops.INDEX dtypes.bf16 [19, 30]
32 Ops.CAST dtypes.f32 [31] dtypes.f32
33 Ops.CONST dtypes.weakint [] 8192
34 Ops.PARAM dtypes.bf16 [33] ParamArg(slot=2, dtype=dtypes.bf16)
35 Ops.MUL dtypes.weakint [15, 20]
36 Ops.ADD dtypes.weakint [35, 29]
37 Ops.INDEX dtypes.bf16 [34, 36]
38 Ops.CAST dtypes.f32 [37] dtypes.f32
39 Ops.MUL dtypes.f32 [32, 38]
40 Ops.REDUCE dtypes.f32 [39, 23, 25, 28] (Ops.ADD, 0)
41 Ops.STORE dtypes.void [17, 40]
42 Ops.END dtypes.void [41, 3, 6, 11, 14]
43 Ops.SINK dtypes.void [42] KernelInfo(name="r_8_32_12_4_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(12, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

