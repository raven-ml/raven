# tolk D89 D212 golden=5b6173c880a824022cbe76138cd4c565
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 9)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 6)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.MUL dtypes.weakint [11, 2]
13 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 4)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 256
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.RANGE dtypes.weakint [8] (AxisType.GLOBAL, 2)
18 Ops.CONST dtypes.weakint [] 16
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.LOCAL, 8)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.MUL dtypes.weakint [21, 8]
23 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.MUL dtypes.weakint [24, 2]
26 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [16, 27]
29 Ops.INDEX dtypes.f32 [1, 28]
30 Ops.CONST dtypes.weakint [] 16384
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=1, dtype=dtypes.f32)
32 Ops.CONST dtypes.weakint [] 64
33 Ops.MUL dtypes.weakint [14, 32]
34 Ops.RANGE dtypes.weakint [18] (AxisType.WEAK, 0)
35 Ops.MUL dtypes.weakint [34, 2]
36 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 7)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.ADD dtypes.weakint [33, 37]
39 Ops.INDEX dtypes.f32 [31, 38]
40 Ops.PARAM dtypes.f32 [30] ParamArg(slot=2, dtype=dtypes.f32)
41 Ops.MUL dtypes.weakint [27, 32]
42 Ops.ADD dtypes.weakint [41, 37]
43 Ops.INDEX dtypes.f32 [40, 42]
44 Ops.MUL dtypes.f32 [39, 43]
45 Ops.REDUCE dtypes.f32 [44, 34, 36] (Ops.ADD, 0)
46 Ops.STORE dtypes.void [29, 45]
47 Ops.END dtypes.void [46, 3, 6, 10, 13, 17, 20, 23, 26]
48 Ops.SINK dtypes.void [47] KernelInfo(name="r_4_2_16_8_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), optimized=True)

