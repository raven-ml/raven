# tolk D141 D280 golden=57397dc67fc52d6a1f1781a57920e9c5
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.AFTER dtypes.f32 [1, 6]
8 Ops.INDEX dtypes.f32 [7, 6]
9 Ops.CONST dtypes.weakfloat [] 0.0
10 Ops.CONST dtypes.weakint [] 64
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 1
13 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 1)
14 Ops.CONST dtypes.weakint [] 2
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 4)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.CONST dtypes.weakint [] 32
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
21 Ops.CONST dtypes.weakint [] 8
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.ADD dtypes.weakint [19, 24]
26 Ops.INDEX dtypes.f32 [11, 25]
27 Ops.CONST dtypes.weakint [] 1024
28 Ops.PARAM dtypes.f32 [27] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [6, 18]
30 Ops.ADD dtypes.weakint [29, 24]
31 Ops.CONST dtypes.weakint [] 512
32 Ops.MUL dtypes.weakint [17, 31]
33 Ops.ADD dtypes.weakint [30, 32]
34 Ops.INDEX dtypes.f32 [28, 33]
35 Ops.MUL dtypes.f32 [26, 34]
36 Ops.REDUCE dtypes.f32 [35, 20, 23] (Ops.ADD, 0)
37 Ops.CMPLT dtypes.bool [9, 36]
38 Ops.WHERE dtypes.f32 [37, 36, 9]
39 Ops.PARAM dtypes.f32 [14] ParamArg(slot=3, dtype=dtypes.f32)
40 Ops.INDEX dtypes.f32 [39, 17]
41 Ops.MUL dtypes.f32 [38, 40]
42 Ops.REDUCE dtypes.f32 [41, 13, 16] (Ops.ADD, 0)
43 Ops.STORE dtypes.void [8, 42]
44 Ops.END dtypes.void [43, 3, 5]
45 Ops.SINK dtypes.void [44] KernelInfo(name="r_4_4_4_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.UPCAST))), optimized=True)

