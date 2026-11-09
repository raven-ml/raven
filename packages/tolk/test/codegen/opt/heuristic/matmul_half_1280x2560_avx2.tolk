# tolk D141 golden=f706ceb417b18ea7919f1a1436bf97b6
0 Ops.CONST dtypes.weakint [] 3276800
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 320
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2560
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 160
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 2)
12 Ops.CONST dtypes.weakint [] 16
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 81920
19 Ops.PARAM dtypes.f16 [18] ParamArg(slot=1, dtype=dtypes.f16)
20 Ops.CONST dtypes.weakint [] 64
21 Ops.MUL dtypes.weakint [7, 20]
22 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.INDEX dtypes.f16 [19, 23]
25 Ops.CONST dtypes.weakint [] 163840
26 Ops.PARAM dtypes.f16 [25] ParamArg(slot=2, dtype=dtypes.f16)
27 Ops.MUL dtypes.weakint [22, 8]
28 Ops.ADD dtypes.weakint [27, 15]
29 Ops.INDEX dtypes.f16 [26, 28]
30 Ops.MUL dtypes.f16 [24, 29]
31 Ops.CAST dtypes.f32 [30] dtypes.f32
32 Ops.REDUCE dtypes.f32 [31, 22] (Ops.ADD, 0)
33 Ops.STORE dtypes.void [17, 32]
34 Ops.END dtypes.void [33, 3, 6, 11, 14]
35 Ops.SINK dtypes.void [34] KernelInfo(name="r_320_160_16_4_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))), optimized=True)

