# tolk D141 D188 golden=a2ac4bd48fd954200fcf635dbcbb09f0
0 Ops.CONST dtypes.weakint [] 384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 3
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 2
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.MUL dtypes.weakint [11, 10]
13 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 5)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 8
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.RANGE dtypes.weakint [15] (AxisType.UPCAST, 3)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.ADD dtypes.weakint [9, 18]
20 Ops.INDEX dtypes.f32 [1, 19]
21 Ops.CONST dtypes.weakint [] 3072
22 Ops.PARAM dtypes.bf16 [21] ParamArg(slot=1, dtype=dtypes.bf16)
23 Ops.CONST dtypes.weakint [] 256
24 Ops.MUL dtypes.weakint [7, 23]
25 Ops.CONST dtypes.weakint [] 64
26 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 0)
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [24, 29]
31 Ops.INDEX dtypes.bf16 [22, 30]
32 Ops.CAST dtypes.f32 [31] dtypes.f32
33 Ops.CONST dtypes.weakint [] 8192
34 Ops.PARAM dtypes.bf16 [33] ParamArg(slot=2, dtype=dtypes.bf16)
35 Ops.MUL dtypes.weakint [18, 23]
36 Ops.ADD dtypes.weakint [35, 29]
37 Ops.INDEX dtypes.bf16 [34, 36]
38 Ops.CAST dtypes.f32 [37] dtypes.f32
39 Ops.MUL dtypes.f32 [32, 38]
40 Ops.REDUCE dtypes.f32 [39, 26, 28] (Ops.ADD, 0)
41 Ops.STORE dtypes.void [20, 40]
42 Ops.END dtypes.void [41, 3, 6, 11, 13, 17]
43 Ops.SINK dtypes.void [42] KernelInfo(name="r_3_2_8_4_2_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(4, AxisType.UPCAST))), optimized=True)

