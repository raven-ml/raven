# tolk D212 golden=ffdc7c97816681802575e2d5809dfd8f
0 Ops.CONST dtypes.weakint [] 1572864
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 10)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 7)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 5)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.CONST dtypes.weakint [] 768
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.CONST dtypes.weakint [] 6
19 Ops.RANGE dtypes.weakint [18] (AxisType.GLOBAL, 3)
20 Ops.MUL dtypes.weakint [19, 2]
21 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 9)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.MUL dtypes.weakint [22, 8]
24 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 6)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [25, 12]
27 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 4)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.ADD dtypes.weakint [17, 28]
30 Ops.RANGE dtypes.weakint [8] (AxisType.GLOBAL, 1)
31 Ops.CONST dtypes.weakint [] 786432
32 Ops.MUL dtypes.weakint [30, 31]
33 Ops.ADD dtypes.weakint [29, 32]
34 Ops.INDEX dtypes.f32 [1, 33]
35 Ops.CONST dtypes.weakint [] 32768
36 Ops.PARAM dtypes.f32 [35] ParamArg(slot=1, dtype=dtypes.f32)
37 Ops.MUL dtypes.weakint [15, 2]
38 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
39 Ops.MUL dtypes.weakint [38, 12]
40 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 8)
41 Ops.ADD dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [37, 41]
43 Ops.CONST dtypes.weakint [] 16384
44 Ops.MUL dtypes.weakint [30, 43]
45 Ops.ADD dtypes.weakint [42, 44]
46 Ops.INDEX dtypes.f32 [36, 45]
47 Ops.CONST dtypes.weakint [] 24576
48 Ops.PARAM dtypes.f32 [47] ParamArg(slot=2, dtype=dtypes.f32)
49 Ops.MUL dtypes.weakint [41, 16]
50 Ops.ADD dtypes.weakint [49, 28]
51 Ops.CONST dtypes.weakint [] 12288
52 Ops.MUL dtypes.weakint [30, 51]
53 Ops.ADD dtypes.weakint [50, 52]
54 Ops.INDEX dtypes.f32 [48, 53]
55 Ops.MUL dtypes.f32 [46, 54]
56 Ops.REDUCE dtypes.f32 [55, 38, 40] (Ops.ADD, 0)
57 Ops.STORE dtypes.void [34, 56]
58 Ops.END dtypes.void [57, 30, 3, 6, 10, 14, 19, 21, 24, 27]
59 Ops.SINK dtypes.void [58] KernelInfo(name="r_2_16_6_16_8_4_4_2_2_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=7, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL))), optimized=True)

