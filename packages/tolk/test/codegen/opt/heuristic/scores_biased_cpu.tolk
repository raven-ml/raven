# tolk D141 golden=8c1b5fb2d012de41c259c545d580b831
0 Ops.CONST dtypes.weakint [] 16777216
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 256
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.CONST dtypes.weakint [] 32
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 4)
7 Ops.CONST dtypes.weakint [] 2
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.RANGE dtypes.weakint [7] (AxisType.UPCAST, 7)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 5)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [4, 14]
16 Ops.CONST dtypes.weakint [] 1
17 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 2)
18 Ops.MUL dtypes.weakint [17, 11]
19 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 6)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.CONST dtypes.weakint [] 65536
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.ADD dtypes.weakint [15, 22]
24 Ops.CONST dtypes.weakint [] 64
25 Ops.RANGE dtypes.weakint [24] (AxisType.WEAK, 1)
26 Ops.CONST dtypes.weakint [] 262144
27 Ops.MUL dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [23, 27]
29 Ops.INDEX dtypes.f32 [1, 28]
30 Ops.CONST dtypes.weakint [] 1048576
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=1, dtype=dtypes.f32)
32 Ops.CONST dtypes.weakint [] 16
33 Ops.MUL dtypes.weakint [20, 32]
34 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 0)
35 Ops.MUL dtypes.weakint [34, 32]
36 Ops.RANGE dtypes.weakint [32] (AxisType.UPCAST, 8)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.ADD dtypes.weakint [33, 37]
39 Ops.MUL dtypes.weakint [3, 24]
40 Ops.ADD dtypes.weakint [38, 39]
41 Ops.CONST dtypes.weakint [] 16384
42 Ops.MUL dtypes.weakint [25, 41]
43 Ops.ADD dtypes.weakint [40, 42]
44 Ops.INDEX dtypes.f32 [31, 43]
45 Ops.PARAM dtypes.f32 [30] ParamArg(slot=2, dtype=dtypes.f32)
46 Ops.MUL dtypes.weakint [14, 24]
47 Ops.ADD dtypes.weakint [38, 46]
48 Ops.ADD dtypes.weakint [47, 42]
49 Ops.INDEX dtypes.f32 [45, 48]
50 Ops.MUL dtypes.f32 [44, 49]
51 Ops.REDUCE dtypes.f32 [50, 34, 36] (Ops.ADD, 0)
52 Ops.PARAM dtypes.f32 [41] ParamArg(slot=3, dtype=dtypes.f32)
53 Ops.MUL dtypes.weakint [25, 2]
54 Ops.ADD dtypes.weakint [53, 14]
55 Ops.INDEX dtypes.f32 [52, 54]
56 Ops.ADD dtypes.f32 [51, 55]
57 Ops.PARAM dtypes.f32 [41] ParamArg(slot=4, dtype=dtypes.f32)
58 Ops.ADD dtypes.weakint [53, 3]
59 Ops.INDEX dtypes.f32 [57, 58]
60 Ops.PARAM dtypes.f32 [41] ParamArg(slot=5, dtype=dtypes.f32)
61 Ops.INDEX dtypes.f32 [60, 58]
62 Ops.MUL dtypes.f32 [59, 61]
63 Ops.ADD dtypes.f32 [56, 62]
64 Ops.STORE dtypes.void [29, 63]
65 Ops.END dtypes.void [64, 25, 17, 19, 3, 6, 9, 13]
66 Ops.SINK dtypes.void [65] KernelInfo(name="r_64_256_32_4_4_2_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=3, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(0, AxisType.UPCAST))), optimized=True)

