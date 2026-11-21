# tolk D141 D188 D502 golden=891fe40cb84d09a8f39f58aea81ee2f1
0 Ops.CONST dtypes.weakint [] 1802240
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 4)
4 Ops.CONST dtypes.weakint [] 16
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 110
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 64
13 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 3)
14 Ops.CONST dtypes.weakint [] 4
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 6)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.MUL dtypes.weakint [17, 8]
19 Ops.ADD dtypes.weakint [18, 7]
20 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 2)
21 Ops.CONST dtypes.weakint [] 28160
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.ADD dtypes.weakint [19, 22]
24 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
25 Ops.CONST dtypes.weakint [] 112640
26 Ops.MUL dtypes.weakint [24, 25]
27 Ops.ADD dtypes.weakint [23, 26]
28 Ops.CONST dtypes.bool [] Invalid
29 Ops.WHERE dtypes.weakint [11, 27, 28]
30 Ops.INDEX dtypes.f32 [1, 29]
31 Ops.CONST dtypes.weakint [] 262144
32 Ops.PARAM dtypes.f32 [31] ParamArg(slot=1, dtype=dtypes.f32)
33 Ops.MUL dtypes.weakint [17, 12]
34 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
35 Ops.MUL dtypes.weakint [34, 14]
36 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 7)
37 Ops.ADD dtypes.weakint [35, 36]
38 Ops.ADD dtypes.weakint [33, 37]
39 Ops.CONST dtypes.weakint [] 16384
40 Ops.MUL dtypes.weakint [24, 39]
41 Ops.ADD dtypes.weakint [38, 40]
42 Ops.INDEX dtypes.f32 [32, 41]
43 Ops.PARAM dtypes.f32 [21] ParamArg(slot=2, dtype=dtypes.f32)
44 Ops.MUL dtypes.weakint [20, 8]
45 Ops.CONST dtypes.weakint [] 109
46 Ops.WHERE dtypes.weakint [9, 7, 45]
47 Ops.ADD dtypes.weakint [44, 46]
48 Ops.CONST dtypes.weakint [] 440
49 Ops.MUL dtypes.weakint [37, 48]
50 Ops.ADD dtypes.weakint [47, 49]
51 Ops.INDEX dtypes.f32 [43, 50]
52 Ops.MUL dtypes.f32 [42, 51]
53 Ops.REDUCE dtypes.f32 [52, 34, 36] (Ops.ADD, 0)
54 Ops.PARAM dtypes.f32 [14] ParamArg(slot=3, dtype=dtypes.f32)
55 Ops.INDEX dtypes.f32 [54, 20]
56 Ops.ADD dtypes.f32 [53, 55]
57 Ops.STORE dtypes.void [30, 56]
58 Ops.END dtypes.void [57, 24, 20, 13, 16, 3, 6]
59 Ops.SINK dtypes.void [58] KernelInfo(name="r_16_4_64_7_16_4_16_4", applied_opts=(Opt(op=OptOps.PADTO, axis=3, arg=4), Opt(op=OptOps.SPLIT, axis=3, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST))), optimized=True)

