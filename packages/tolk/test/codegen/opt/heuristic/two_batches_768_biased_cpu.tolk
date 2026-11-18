# tolk D141 golden=038cbfeaf5c441c209dc20fc5b6733a8
0 Ops.CONST dtypes.weakint [] 1572864
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1024
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 768
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 24
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
17 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 1)
18 Ops.CONST dtypes.weakint [] 786432
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.ADD dtypes.weakint [16, 19]
21 Ops.INDEX dtypes.f32 [1, 20]
22 Ops.CONST dtypes.weakint [] 32768
23 Ops.PARAM dtypes.f32 [22] ParamArg(slot=1, dtype=dtypes.f32)
24 Ops.MUL dtypes.weakint [3, 12]
25 Ops.CONST dtypes.weakint [] 1
26 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 0)
27 Ops.MUL dtypes.weakint [26, 12]
28 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 6)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [24, 29]
31 Ops.CONST dtypes.weakint [] 16384
32 Ops.MUL dtypes.weakint [17, 31]
33 Ops.ADD dtypes.weakint [30, 32]
34 Ops.INDEX dtypes.f32 [23, 33]
35 Ops.CONST dtypes.weakint [] 24576
36 Ops.PARAM dtypes.f32 [35] ParamArg(slot=2, dtype=dtypes.f32)
37 Ops.MUL dtypes.weakint [29, 4]
38 Ops.ADD dtypes.weakint [37, 15]
39 Ops.CONST dtypes.weakint [] 12288
40 Ops.MUL dtypes.weakint [17, 39]
41 Ops.ADD dtypes.weakint [38, 40]
42 Ops.INDEX dtypes.f32 [36, 41]
43 Ops.MUL dtypes.f32 [34, 42]
44 Ops.REDUCE dtypes.f32 [43, 26, 28] (Ops.ADD, 0)
45 Ops.PARAM dtypes.f32 [18] ParamArg(slot=3, dtype=dtypes.f32)
46 Ops.INDEX dtypes.f32 [45, 16]
47 Ops.ADD dtypes.f32 [44, 46]
48 Ops.STORE dtypes.void [21, 47]
49 Ops.END dtypes.void [48, 17, 3, 7, 10, 14]
50 Ops.SINK dtypes.void [49] KernelInfo(name="r_2_1024_24_16_2_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST))), optimized=True)

