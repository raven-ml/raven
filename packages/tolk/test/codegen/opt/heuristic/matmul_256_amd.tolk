# tolk D89 golden=49e3d3a394b7eb7238d2134f7192bed0
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 8)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 4)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 256
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.RANGE dtypes.weakint [4] (AxisType.GLOBAL, 2)
18 Ops.MUL dtypes.weakint [17, 2]
19 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 9)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.MUL dtypes.weakint [20, 4]
22 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.MUL dtypes.weakint [23, 11]
25 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 3)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.ADD dtypes.weakint [16, 26]
28 Ops.INDEX dtypes.f32 [1, 27]
29 Ops.CONST dtypes.weakint [] 16384
30 Ops.PARAM dtypes.f32 [29] ParamArg(slot=1, dtype=dtypes.f32)
31 Ops.CONST dtypes.weakint [] 64
32 Ops.MUL dtypes.weakint [14, 31]
33 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
34 Ops.MUL dtypes.weakint [33, 11]
35 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 7)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.ADD dtypes.weakint [32, 36]
38 Ops.INDEX dtypes.f32 [30, 37]
39 Ops.PARAM dtypes.f32 [29] ParamArg(slot=2, dtype=dtypes.f32)
40 Ops.MUL dtypes.weakint [36, 15]
41 Ops.ADD dtypes.weakint [40, 26]
42 Ops.INDEX dtypes.f32 [39, 41]
43 Ops.MUL dtypes.f32 [38, 42]
44 Ops.REDUCE dtypes.f32 [43, 33, 35] (Ops.ADD, 0)
45 Ops.STORE dtypes.void [28, 44]
46 Ops.END dtypes.void [45, 3, 6, 9, 13, 17, 19, 22, 25]
47 Ops.SINK dtypes.void [46] KernelInfo(name="r_16_2_2_16_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL))), optimized=True)

