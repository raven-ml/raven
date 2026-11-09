# tolk D141 golden=a8116307d7697dd30af93d6a621d7d71
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 128
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 16
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 2880
14 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0)
16 Ops.INDEX dtypes.f32 [14, 15]
17 Ops.PARAM dtypes.f32 [13] ParamArg(slot=3, dtype=dtypes.f32)
18 Ops.INDEX dtypes.f32 [17, 15]
19 Ops.MUL dtypes.f32 [16, 18]
20 Ops.CONST dtypes.weakint [] 11796480
21 Ops.PARAM dtypes.bf16 [20] ParamArg(slot=4, dtype=dtypes.bf16)
22 Ops.MUL dtypes.weakint [15, 0]
23 Ops.ADD dtypes.weakint [22, 11]
24 Ops.INDEX dtypes.bf16 [21, 23]
25 Ops.CAST dtypes.f32 [24] dtypes.f32
26 Ops.MUL dtypes.f32 [19, 25]
27 Ops.REDUCE dtypes.f32 [26, 15] (Ops.ADD, 0)
28 Ops.CONST dtypes.weakint [] 1
29 Ops.PARAM dtypes.f32 [28] ParamArg(slot=2, dtype=dtypes.f32)
30 Ops.CONST dtypes.weakint [] 0
31 Ops.INDEX dtypes.f32 [29, 30]
32 Ops.RECIP dtypes.f32 [31]
33 Ops.MUL dtypes.f32 [27, 32]
34 Ops.STORE dtypes.void [12, 33]
35 Ops.END dtypes.void [34, 3, 6, 10]
36 Ops.SINK dtypes.void [35] KernelInfo(name="r_128_16_2_2880", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST))), optimized=True)

