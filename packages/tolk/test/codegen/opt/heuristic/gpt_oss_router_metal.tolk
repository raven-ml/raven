# tolk D89 golden=41617e40accd3e6fff13010437097e15
0 Ops.CONST dtypes.weakint [] 32
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.CONST dtypes.weakint [] 8
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.RANGE dtypes.weakint [7] (AxisType.LOCAL, 2)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.INDEX dtypes.f32 [1, 10]
12 Ops.CONST dtypes.weakint [] 2880
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.CONST dtypes.weakint [] 45
15 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 0)
16 Ops.CONST dtypes.weakint [] 64
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.RANGE dtypes.weakint [16] (AxisType.LOCAL, 4)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.INDEX dtypes.f32 [13, 19]
21 Ops.PARAM dtypes.f32 [12] ParamArg(slot=3, dtype=dtypes.f32)
22 Ops.INDEX dtypes.f32 [21, 19]
23 Ops.MUL dtypes.f32 [20, 22]
24 Ops.CONST dtypes.weakint [] 92160
25 Ops.PARAM dtypes.bf16 [24] ParamArg(slot=4, dtype=dtypes.bf16)
26 Ops.MUL dtypes.weakint [19, 0]
27 Ops.ADD dtypes.weakint [26, 10]
28 Ops.INDEX dtypes.bf16 [25, 27]
29 Ops.CAST dtypes.f32 [28] dtypes.f32
30 Ops.MUL dtypes.f32 [23, 29]
31 Ops.REDUCE dtypes.f32 [30, 15, 18] (Ops.ADD, 0)
32 Ops.CONST dtypes.weakint [] 1
33 Ops.PARAM dtypes.f32 [32] ParamArg(slot=2, dtype=dtypes.f32)
34 Ops.CONST dtypes.weakint [] 0
35 Ops.INDEX dtypes.f32 [33, 34]
36 Ops.RECIP dtypes.f32 [35]
37 Ops.MUL dtypes.f32 [31, 36]
38 Ops.STORE dtypes.void [11, 37]
39 Ops.END dtypes.void [38, 3, 5, 9]
40 Ops.SINK dtypes.void [39] KernelInfo(name="r_2_8_64_2_45", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(64, AxisType.LOCAL))), optimized=True)

