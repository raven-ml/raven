# tolk D141 D188 golden=8c1b5fb2d012de41c259c545d580b831
0 Ops.CONST dtypes.weakint [] 16777216
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 256
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 64
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 4)
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 5)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 2)
18 Ops.CONST dtypes.weakint [] 65536
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.ADD dtypes.weakint [16, 19]
21 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 1)
22 Ops.CONST dtypes.weakint [] 262144
23 Ops.MUL dtypes.weakint [21, 22]
24 Ops.ADD dtypes.weakint [20, 23]
25 Ops.INDEX dtypes.f32 [1, 24]
26 Ops.CONST dtypes.weakint [] 1048576
27 Ops.PARAM dtypes.f32 [26] ParamArg(slot=1, dtype=dtypes.f32)
28 Ops.CONST dtypes.weakint [] 16
29 Ops.MUL dtypes.weakint [17, 28]
30 Ops.CONST dtypes.weakint [] 1
31 Ops.RANGE dtypes.weakint [30] (AxisType.WEAK, 0)
32 Ops.MUL dtypes.weakint [31, 28]
33 Ops.RANGE dtypes.weakint [28] (AxisType.UPCAST, 7)
34 Ops.ADD dtypes.weakint [32, 33]
35 Ops.ADD dtypes.weakint [29, 34]
36 Ops.MUL dtypes.weakint [7, 10]
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.CONST dtypes.weakint [] 16384
39 Ops.MUL dtypes.weakint [21, 38]
40 Ops.ADD dtypes.weakint [37, 39]
41 Ops.INDEX dtypes.f32 [27, 40]
42 Ops.PARAM dtypes.f32 [26] ParamArg(slot=2, dtype=dtypes.f32)
43 Ops.MUL dtypes.weakint [15, 10]
44 Ops.ADD dtypes.weakint [35, 43]
45 Ops.ADD dtypes.weakint [44, 39]
46 Ops.INDEX dtypes.f32 [42, 45]
47 Ops.MUL dtypes.f32 [41, 46]
48 Ops.REDUCE dtypes.f32 [47, 31, 33] (Ops.ADD, 0)
49 Ops.PARAM dtypes.f32 [38] ParamArg(slot=3, dtype=dtypes.f32)
50 Ops.MUL dtypes.weakint [21, 8]
51 Ops.ADD dtypes.weakint [50, 15]
52 Ops.INDEX dtypes.f32 [49, 51]
53 Ops.ADD dtypes.f32 [48, 52]
54 Ops.PARAM dtypes.f32 [38] ParamArg(slot=4, dtype=dtypes.f32)
55 Ops.ADD dtypes.weakint [50, 7]
56 Ops.INDEX dtypes.f32 [54, 55]
57 Ops.PARAM dtypes.f32 [38] ParamArg(slot=5, dtype=dtypes.f32)
58 Ops.INDEX dtypes.f32 [57, 55]
59 Ops.MUL dtypes.f32 [56, 58]
60 Ops.ADD dtypes.f32 [53, 59]
61 Ops.STORE dtypes.void [25, 60]
62 Ops.END dtypes.void [61, 21, 17, 3, 6, 11, 14]
63 Ops.SINK dtypes.void [62] KernelInfo(name="r_64_4_32_64_4_8_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=3, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(0, AxisType.UPCAST))), optimized=True)

