# tolk D141 golden=a83b2186a9ecb936cf48c7f00e479fe0
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.CONST dtypes.weakint [] 64
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.CONST dtypes.weakint [] 16
10 Ops.RANGE dtypes.weakint [9] (AxisType.WEAK, 2)
11 Ops.CONST dtypes.weakint [] 4
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 4)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [8, 14]
16 Ops.INDEX dtypes.f32 [1, 15]
17 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
18 Ops.RANGE dtypes.weakint [9] (AxisType.WEAK, 0)
19 Ops.MUL dtypes.weakint [18, 11]
20 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 5)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.ADD dtypes.weakint [8, 21]
23 Ops.INDEX dtypes.bf16 [17, 22]
24 Ops.CAST dtypes.f32 [23] dtypes.f32
25 Ops.PARAM dtypes.i8 [0] ParamArg(slot=2, dtype=dtypes.i8)
26 Ops.MUL dtypes.weakint [21, 7]
27 Ops.ADD dtypes.weakint [26, 14]
28 Ops.INDEX dtypes.i8 [25, 27]
29 Ops.CAST dtypes.f32 [28] dtypes.f32
30 Ops.MUL dtypes.f32 [24, 29]
31 Ops.REDUCE dtypes.f32 [30, 18, 20] (Ops.ADD, 0)
32 Ops.STORE dtypes.void [16, 31]
33 Ops.END dtypes.void [32, 3, 5, 10, 13]
34 Ops.SINK dtypes.void [33] KernelInfo(name="r_8_16_8_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

