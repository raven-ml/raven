# tolk D89 golden=2934aeb200a5c8250e3a2375cba5547f
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 8)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 2]
9 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 6)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.CONST dtypes.weakint [] 64
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.CONST dtypes.weakint [] 1
17 Ops.RANGE dtypes.weakint [16] (AxisType.GLOBAL, 2)
18 Ops.CONST dtypes.weakint [] 8
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.LOCAL, 9)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.MUL dtypes.weakint [21, 2]
23 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.MUL dtypes.weakint [24, 4]
26 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [15, 27]
29 Ops.INDEX dtypes.f32 [1, 28]
30 Ops.CONST dtypes.weakint [] 65536
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=1, dtype=dtypes.f32)
32 Ops.CONST dtypes.weakint [] 1024
33 Ops.MUL dtypes.weakint [13, 32]
34 Ops.CONST dtypes.weakint [] 256
35 Ops.RANGE dtypes.weakint [34] (AxisType.WEAK, 0)
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.ADD dtypes.weakint [33, 38]
40 Ops.INDEX dtypes.f32 [31, 39]
41 Ops.PARAM dtypes.f32 [30] ParamArg(slot=2, dtype=dtypes.f32)
42 Ops.MUL dtypes.weakint [38, 14]
43 Ops.ADD dtypes.weakint [42, 27]
44 Ops.INDEX dtypes.f32 [41, 43]
45 Ops.MUL dtypes.f32 [40, 44]
46 Ops.REDUCE dtypes.f32 [45, 35, 37] (Ops.ADD, 0)
47 Ops.STORE dtypes.void [29, 46]
48 Ops.END dtypes.void [47, 3, 6, 9, 12, 17, 20, 23, 26]
49 Ops.SINK dtypes.void [48] KernelInfo(name="r_2_4_8_4_4_2_2_256_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL))), optimized=True)

