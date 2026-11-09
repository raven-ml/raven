# tolk D141 golden=57397dc67fc52d6a1f1781a57920e9c5
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.INDEX dtypes.f32 [1, 6]
8 Ops.CONST dtypes.weakfloat [] 0.0
9 Ops.CONST dtypes.weakint [] 64
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 1
12 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 1)
13 Ops.CONST dtypes.weakint [] 2
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 4)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 32
18 Ops.MUL dtypes.weakint [16, 17]
19 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
20 Ops.CONST dtypes.weakint [] 8
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.RANGE dtypes.weakint [20] (AxisType.UPCAST, 5)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.ADD dtypes.weakint [18, 23]
25 Ops.INDEX dtypes.f32 [10, 24]
26 Ops.CONST dtypes.weakint [] 1024
27 Ops.PARAM dtypes.f32 [26] ParamArg(slot=2, dtype=dtypes.f32)
28 Ops.MUL dtypes.weakint [6, 17]
29 Ops.ADD dtypes.weakint [28, 23]
30 Ops.CONST dtypes.weakint [] 512
31 Ops.MUL dtypes.weakint [16, 30]
32 Ops.ADD dtypes.weakint [29, 31]
33 Ops.INDEX dtypes.f32 [27, 32]
34 Ops.MUL dtypes.f32 [25, 33]
35 Ops.REDUCE dtypes.f32 [34, 19, 22] (Ops.ADD, 0)
36 Ops.MAX dtypes.f32 [8, 35]
37 Ops.PARAM dtypes.f32 [13] ParamArg(slot=3, dtype=dtypes.f32)
38 Ops.INDEX dtypes.f32 [37, 16]
39 Ops.MUL dtypes.f32 [36, 38]
40 Ops.REDUCE dtypes.f32 [39, 12, 15] (Ops.ADD, 0)
41 Ops.STORE dtypes.void [7, 40]
42 Ops.END dtypes.void [41, 3, 5]
43 Ops.SINK dtypes.void [42] KernelInfo(name="r_4_4_4_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.UPCAST))), optimized=True)

