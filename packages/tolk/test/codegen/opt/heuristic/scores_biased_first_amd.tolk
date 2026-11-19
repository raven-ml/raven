# tolk D212 D495 golden=52b0585c3b1a37bcd88ead0270bf31d8
0 Ops.CONST dtypes.weakint [] 16777216
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 3)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 11)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 8)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.MUL dtypes.weakint [11, 2]
13 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 6)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 256
16 Ops.MUL dtypes.weakint [14, 15]
17 Ops.RANGE dtypes.weakint [8] (AxisType.GLOBAL, 4)
18 Ops.CONST dtypes.weakint [] 16
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.LOCAL, 10)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.MUL dtypes.weakint [21, 8]
23 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 7)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.MUL dtypes.weakint [24, 2]
26 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 5)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [16, 27]
29 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
30 Ops.CONST dtypes.weakint [] 65536
31 Ops.MUL dtypes.weakint [29, 30]
32 Ops.ADD dtypes.weakint [28, 31]
33 Ops.CONST dtypes.weakint [] 64
34 Ops.RANGE dtypes.weakint [33] (AxisType.GLOBAL, 1)
35 Ops.CONST dtypes.weakint [] 262144
36 Ops.MUL dtypes.weakint [34, 35]
37 Ops.ADD dtypes.weakint [32, 36]
38 Ops.INDEX dtypes.f32 [1, 37]
39 Ops.CONST dtypes.weakint [] 1048576
40 Ops.PARAM dtypes.f32 [39] ParamArg(slot=1, dtype=dtypes.f32)
41 Ops.MUL dtypes.weakint [29, 18]
42 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
43 Ops.MUL dtypes.weakint [42, 2]
44 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 9)
45 Ops.ADD dtypes.weakint [43, 44]
46 Ops.ADD dtypes.weakint [41, 45]
47 Ops.MUL dtypes.weakint [14, 33]
48 Ops.ADD dtypes.weakint [46, 47]
49 Ops.CONST dtypes.weakint [] 16384
50 Ops.MUL dtypes.weakint [34, 49]
51 Ops.ADD dtypes.weakint [48, 50]
52 Ops.INDEX dtypes.f32 [40, 51]
53 Ops.PARAM dtypes.f32 [39] ParamArg(slot=2, dtype=dtypes.f32)
54 Ops.MUL dtypes.weakint [27, 33]
55 Ops.ADD dtypes.weakint [46, 54]
56 Ops.ADD dtypes.weakint [55, 50]
57 Ops.INDEX dtypes.f32 [53, 56]
58 Ops.MUL dtypes.f32 [52, 57]
59 Ops.REDUCE dtypes.f32 [58, 42, 44] (Ops.ADD, 0)
60 Ops.PARAM dtypes.f32 [49] ParamArg(slot=3, dtype=dtypes.f32)
61 Ops.MUL dtypes.weakint [34, 15]
62 Ops.ADD dtypes.weakint [61, 27]
63 Ops.INDEX dtypes.f32 [60, 62]
64 Ops.ADD dtypes.f32 [59, 63]
65 Ops.PARAM dtypes.f32 [49] ParamArg(slot=4, dtype=dtypes.f32)
66 Ops.INDEX dtypes.f32 [65, 62]
67 Ops.ADD dtypes.f32 [64, 66]
68 Ops.PARAM dtypes.f32 [49] ParamArg(slot=5, dtype=dtypes.f32)
69 Ops.ADD dtypes.weakint [61, 14]
70 Ops.INDEX dtypes.f32 [68, 69]
71 Ops.PARAM dtypes.f32 [49] ParamArg(slot=6, dtype=dtypes.f32)
72 Ops.INDEX dtypes.f32 [71, 69]
73 Ops.MUL dtypes.f32 [70, 72]
74 Ops.ADD dtypes.f32 [67, 73]
75 Ops.STORE dtypes.void [38, 74]
76 Ops.END dtypes.void [75, 34, 29, 3, 6, 10, 13, 17, 20, 23, 26]
77 Ops.SINK dtypes.void [76] KernelInfo(name="r_64_4_4_2_16_8_4_4_2_2_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=3, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=8, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.LOCAL))), optimized=True)

