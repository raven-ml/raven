# tolk D141 golden=d224bd9cc10621867398dc21b9e13839
0 Ops.CONST dtypes.weakint [] 1802240
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 110
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 55
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 4)
12 Ops.CONST dtypes.weakint [] 2
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 5)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 2)
18 Ops.MUL dtypes.weakint [17, 12]
19 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 7)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.CONST dtypes.weakint [] 28160
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.ADD dtypes.weakint [16, 22]
24 Ops.CONST dtypes.weakint [] 16
25 Ops.RANGE dtypes.weakint [24] (AxisType.WEAK, 1)
26 Ops.CONST dtypes.weakint [] 112640
27 Ops.MUL dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [23, 27]
29 Ops.INDEX dtypes.f32 [1, 28]
30 Ops.CONST dtypes.weakint [] 262144
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=1, dtype=dtypes.f32)
32 Ops.MUL dtypes.weakint [7, 2]
33 Ops.RANGE dtypes.weakint [24] (AxisType.WEAK, 0)
34 Ops.MUL dtypes.weakint [33, 4]
35 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.ADD dtypes.weakint [32, 36]
38 Ops.CONST dtypes.weakint [] 16384
39 Ops.MUL dtypes.weakint [25, 38]
40 Ops.ADD dtypes.weakint [37, 39]
41 Ops.INDEX dtypes.f32 [31, 40]
42 Ops.PARAM dtypes.f32 [21] ParamArg(slot=2, dtype=dtypes.f32)
43 Ops.MUL dtypes.weakint [20, 8]
44 Ops.ADD dtypes.weakint [43, 15]
45 Ops.CONST dtypes.weakint [] 440
46 Ops.MUL dtypes.weakint [36, 45]
47 Ops.ADD dtypes.weakint [44, 46]
48 Ops.INDEX dtypes.f32 [42, 47]
49 Ops.MUL dtypes.f32 [41, 48]
50 Ops.REDUCE dtypes.f32 [49, 33, 35] (Ops.ADD, 0)
51 Ops.STORE dtypes.void [29, 50]
52 Ops.END dtypes.void [51, 25, 17, 19, 3, 6, 11, 14]
53 Ops.SINK dtypes.void [52] KernelInfo(name="r_16_2_64_55_2_4_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=3, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=7, arg=(4, AxisType.UPCAST))), optimized=True)

