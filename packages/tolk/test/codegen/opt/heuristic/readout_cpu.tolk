# tolk D141 golden=f60b8937c09e1fb0266a4e49b6e7d0c6
0 Ops.CONST dtypes.weakint [] 768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 3
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 256
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 8
7 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 3)
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 5)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 16
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 4)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [5, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 196608
19 Ops.PARAM dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.CONST dtypes.weakint [] 1024
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
22 Ops.MUL dtypes.weakint [21, 2]
23 Ops.ADD dtypes.weakint [22, 3]
24 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 1)
25 Ops.CONST dtypes.weakint [] 4
26 Ops.MUL dtypes.weakint [24, 25]
27 Ops.RANGE dtypes.weakint [25] (AxisType.UPCAST, 6)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.CONST dtypes.weakint [] 3072
30 Ops.MUL dtypes.weakint [28, 29]
31 Ops.ADD dtypes.weakint [23, 30]
32 Ops.INDEX dtypes.f32 [19, 31]
33 Ops.CONST dtypes.weakint [] 16777216
34 Ops.PARAM dtypes.f32 [33] ParamArg(slot=2, dtype=dtypes.f32)
35 Ops.MUL dtypes.weakint [28, 4]
36 Ops.ADD dtypes.weakint [35, 15]
37 Ops.CONST dtypes.weakint [] 16384
38 Ops.MUL dtypes.weakint [21, 37]
39 Ops.ADD dtypes.weakint [36, 38]
40 Ops.INDEX dtypes.f32 [34, 39]
41 Ops.ADD dtypes.f32 [32, 40]
42 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
43 Ops.MUL dtypes.f32 [41, 42]
44 Ops.EXP2 dtypes.f32 [43]
45 Ops.REDUCE dtypes.f32 [44, 21, 24, 27] (Ops.ADD, 0)
46 Ops.STORE dtypes.void [17, 45]
47 Ops.END dtypes.void [46, 3, 7, 10, 14]
48 Ops.SINK dtypes.void [47] KernelInfo(name="r_3_8_16_2_1024_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(4, AxisType.UPCAST))), optimized=True)

