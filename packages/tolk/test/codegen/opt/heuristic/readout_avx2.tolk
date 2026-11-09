# tolk D141 golden=f60b8937c09e1fb0266a4e49b6e7d0c6
0 Ops.CONST dtypes.weakint [] 768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 3
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 256
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 3)
12 Ops.MUL dtypes.weakint [11, 10]
13 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 4)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [9, 14]
16 Ops.INDEX dtypes.f32 [1, 15]
17 Ops.CONST dtypes.weakint [] 196608
18 Ops.PARAM dtypes.f32 [17] ParamArg(slot=1, dtype=dtypes.f32)
19 Ops.CONST dtypes.weakint [] 1024
20 Ops.RANGE dtypes.weakint [19] (AxisType.WEAK, 0)
21 Ops.MUL dtypes.weakint [20, 4]
22 Ops.ADD dtypes.weakint [21, 7]
23 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 1)
24 Ops.CONST dtypes.weakint [] 4
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.RANGE dtypes.weakint [24] (AxisType.UPCAST, 6)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.CONST dtypes.weakint [] 3072
29 Ops.MUL dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [22, 29]
31 Ops.INDEX dtypes.f32 [18, 30]
32 Ops.CONST dtypes.weakint [] 16777216
33 Ops.PARAM dtypes.f32 [32] ParamArg(slot=2, dtype=dtypes.f32)
34 Ops.MUL dtypes.weakint [27, 8]
35 Ops.ADD dtypes.weakint [34, 14]
36 Ops.CONST dtypes.weakint [] 16384
37 Ops.MUL dtypes.weakint [20, 36]
38 Ops.ADD dtypes.weakint [35, 37]
39 Ops.INDEX dtypes.f32 [33, 38]
40 Ops.ADD dtypes.f32 [31, 39]
41 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
42 Ops.MUL dtypes.f32 [40, 41]
43 Ops.EXP2 dtypes.f32 [42]
44 Ops.REDUCE dtypes.f32 [43, 20, 23, 26] (Ops.ADD, 0)
45 Ops.STORE dtypes.void [16, 44]
46 Ops.END dtypes.void [45, 3, 6, 11, 13]
47 Ops.SINK dtypes.void [46] KernelInfo(name="r_16_16_3_1024_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(3, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

