# tolk D89 D212 golden=6064b340b9c3b9496421e6cc9d0f058f
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 16
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 10)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 2]
9 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 6)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 4)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 256
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.CONST dtypes.weakint [] 8
18 Ops.RANGE dtypes.weakint [17] (AxisType.GLOBAL, 2)
19 Ops.MUL dtypes.weakint [18, 11]
20 Ops.RANGE dtypes.weakint [11] (AxisType.LOCAL, 9)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.MUL dtypes.weakint [21, 2]
23 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.MUL dtypes.weakint [24, 11]
26 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 3)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [16, 27]
29 Ops.INDEX dtypes.f32 [1, 28]
30 Ops.CONST dtypes.weakint [] 32768
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=1, dtype=dtypes.f32)
32 Ops.CONST dtypes.weakint [] 128
33 Ops.MUL dtypes.weakint [14, 32]
34 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
35 Ops.MUL dtypes.weakint [34, 2]
36 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 8)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.MUL dtypes.weakint [37, 11]
39 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 7)
40 Ops.ADD dtypes.weakint [38, 39]
41 Ops.ADD dtypes.weakint [33, 40]
42 Ops.INDEX dtypes.f32 [31, 41]
43 Ops.PARAM dtypes.f32 [30] ParamArg(slot=2, dtype=dtypes.f32)
44 Ops.MUL dtypes.weakint [27, 32]
45 Ops.ADD dtypes.weakint [44, 40]
46 Ops.INDEX dtypes.f32 [43, 45]
47 Ops.MUL dtypes.f32 [42, 46]
48 Ops.REDUCE dtypes.f32 [47, 34, 36, 39] (Ops.ADD, 0)
49 Ops.STORE dtypes.void [29, 48]
50 Ops.END dtypes.void [49, 3, 6, 9, 13, 18, 20, 23, 26]
51 Ops.SINK dtypes.void [50] KernelInfo(name="r_2_8_2_4_16_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL))), optimized=True)

