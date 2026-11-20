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
10 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
11 Ops.CONST dtypes.weakint [] 2
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 5)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [9, 17]
19 Ops.INDEX dtypes.f32 [1, 18]
20 Ops.CONST dtypes.weakint [] 3072
21 Ops.PARAM dtypes.bf16 [20] ParamArg(slot=1, dtype=dtypes.bf16)
22 Ops.CONST dtypes.weakint [] 256
23 Ops.MUL dtypes.weakint [7, 22]
24 Ops.CONST dtypes.weakint [] 64
25 Ops.RANGE dtypes.weakint [24] (AxisType.WEAK, 0)
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.ADD dtypes.weakint [23, 28]
30 Ops.INDEX dtypes.bf16 [21, 29]
31 Ops.CAST dtypes.f32 [30] dtypes.f32
32 Ops.CONST dtypes.weakint [] 8192
33 Ops.PARAM dtypes.bf16 [32] ParamArg(slot=2, dtype=dtypes.bf16)
34 Ops.MUL dtypes.weakint [17, 22]
35 Ops.ADD dtypes.weakint [34, 28]
36 Ops.INDEX dtypes.bf16 [33, 35]
37 Ops.CAST dtypes.f32 [36] dtypes.f32
38 Ops.MUL dtypes.f32 [31, 37]
39 Ops.REDUCE dtypes.f32 [38, 25, 27] (Ops.ADD, 0)
40 Ops.STORE dtypes.void [19, 39]
41 Ops.END dtypes.void [40, 3, 6, 10, 13, 16]
42 Ops.SINK dtypes.void [41] KernelInfo(name="r_3_4_4_4_2_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(4, AxisType.UPCAST))), optimized=True)

