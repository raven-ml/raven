# tolk D89 golden=7dab7c849dff42fb451349799e5b5cb1
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 4096
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 128
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0)
17 Ops.MUL dtypes.weakint [16, 8]
18 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 4)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.INDEX dtypes.f32 [14, 19]
21 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
22 Ops.MUL dtypes.f32 [20, 21]
23 Ops.EXP2 dtypes.f32 [22]
24 Ops.CONST dtypes.weakint [] 4194304
25 Ops.PARAM dtypes.f32 [24] ParamArg(slot=2, dtype=dtypes.f32)
26 Ops.MUL dtypes.weakint [19, 0]
27 Ops.ADD dtypes.weakint [26, 11]
28 Ops.INDEX dtypes.f32 [25, 27]
29 Ops.MUL dtypes.f32 [23, 28]
30 Ops.REDUCE dtypes.f32 [29, 16, 18] (Ops.ADD, 0)
31 Ops.STORE dtypes.void [12, 30]
32 Ops.END dtypes.void [31, 3, 6, 10]
33 Ops.SINK dtypes.void [32] KernelInfo(name="r_16_32_32_2_128", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(32, AxisType.LOCAL))), optimized=True)

